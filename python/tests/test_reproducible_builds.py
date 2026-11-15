"""Reproducible builds: a binary names its source files by paths that do not depend on the host, and a generated source has a content-addressed file.

Two exports of the same kernels used to differ byte for byte: Triton records the path of each source file in the locations of the IR, ``ptxas`` writes
it into the cubin's ``.debug_line`` and ``.nv_debug_ptx_txt``, and that path named the checkout, the virtual environment, or the random scratch
directory a generated kernel's source was written to. So the ``sha256`` of a cubin, and with it the manifest digest, depended on where and in which
process it was built (the machine code, ``sha256_code``, did not). Now the frontend of a compile names a file by its module path
(``tilefish._source.source_name``, through ``tilefish._ir.host_independent_paths``), and ``runtime.load_lowered`` writes a generated source to
``<cache>/generated/<sha256>/tilefish_generated/<name>.py``.

No Triton and no GPU here: ``fake_compile`` stands in for the frontend and the compiler, and the debug sections of its cubin hold the file name its
frontend took from ``triton.compiler.code_generator.get_jit_fn_file_line``, as a real cubin's do. ``test_gpu_repro.py`` checks the same with Triton,
in two processes from two directories.
"""

import hashlib
import importlib
import json
import linecache
import os
import pathlib
import subprocess
import sys
import threading
import warnings

import fake_compile
import fake_jit
import pytest

import tilefish
from tilefish import TilefishCacheWarning, _ir, _source, runtime
from tilefish._elf import parse_elf
from tilefish.aot import build, compile_specs
from tilefish.dims import cdiv

SM90 = 90
TESTS = pathlib.Path(__file__).resolve().parent
SRC = pathlib.Path(tilefish.__file__).resolve().parent.parent

KERNEL_SOURCE = (
    "from fake_jit import jit\n"
    "\n"
    "SCALE = 3.0\n"
    "\n"
    "\n"
    "@jit\n"
    "def kernel(x_ptr, y_ptr, n_elem, BLOCK):\n"
    "    return (n_elem * SCALE, BLOCK)\n"
)

n = tilefish.Dim("n", divisible_by=16)


@tilefish.extern(triton="rb_pkg.kern:kernel", num_warps=4, grid=lambda n: (cdiv(n, 256),))
def rb_extern(x_ptr: tilefish.Tensor[n, tilefish.bf16], y_ptr: tilefish.Out[tilefish.Tensor[n, tilefish.bf16]], n_elem: n, *, BLOCK: tilefish.const = 256): ...


def scale_kernel(factor=2.0):
    M, D = tilefish.Dim("M"), tilefish.Dim("D", 64)

    @tilefish.kernel(name="rb_scale")
    def rb_scale(x: tilefish.Tensor[M, D, tilefish.f32]) -> tilefish.Tensor[M, D, tilefish.f32]:
        return x * factor

    return rb_scale


@pytest.fixture
def fake(tmp_path, monkeypatch):
    compiler = fake_compile.install(monkeypatch)
    monkeypatch.setattr(runtime, "_modules", {})  # a fresh process: no generated module imported yet
    monkeypatch.setattr(runtime, "_gen_root", [])
    monkeypatch.setattr(runtime, "_warned_gen_roots", set())
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    yield compiler
    for name in ("rb_pkg", "rb_pkg.kern"):
        sys.modules.pop(name, None)


def checkout(root):
    """A package ``rb_pkg`` with the kernel module, under ``root`` (a checkout of the same sources somewhere else)."""
    package = root / "rb_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "kern.py").write_text(KERNEL_SOURCE)
    return root


def import_from(root, monkeypatch):
    """Import ``rb_pkg.kern`` from ``root`` as a new process would (nothing of an earlier import is kept)."""
    for name in ("rb_pkg", "rb_pkg.kern"):
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(root))
    importlib.invalidate_caches()
    module = importlib.import_module("rb_pkg.kern")
    assert pathlib.Path(module.__file__).resolve().is_relative_to(root.resolve())
    return module.kernel


def debug_names(cubin):
    """The text of the ``.debug_line`` section: the file name the fake frontend recorded (``line:<name>``)."""
    elf = parse_elf(cubin)
    return elf.data(elf.section(".debug_line")).decode()


# -- the name of a source file -------------------------------------------------------------------------------------------------------------------------


def test_a_source_file_is_named_by_its_module_path_or_its_base_name():
    name = _source.source_name
    assert name("/home/someone/checkout/kernels/rms.py", "kernels.rms") == "kernels/rms.py"
    assert name("/opt/venv/lib/python3.11/site-packages/triton/language/standard.py", "triton.language.standard") == "triton/language/standard.py"
    assert name("/srv/a/pkg/sub/__init__.py", "pkg.sub") == "pkg/sub/__init__.py"
    assert name("relative/pkg/mod.py", "pkg.mod") == "pkg/mod.py"
    assert name("C:\\work\\pkg\\mod.py", "pkg.mod") == "pkg/mod.py"
    # a script, a module imported under a name its path does not end with, no module at all: the base name
    assert name("/home/someone/run.py", "__main__") == "run.py"
    assert name("/tmp/x/tilefish_generated/softmax.py", "tilefish_generated_softmax__0123456789ab") == "softmax.py"
    assert name("/tmp/x/kernels/rms.py", None) == "rms.py"
    assert name("/elsewhere/rms.py", "kernels.rms") == "rms.py"  # the tail is not the module path: nothing of the directory is kept
    assert name("<string>", "m") == "<string>" and name("", "m") == ""  # not a path


def test_a_generated_source_is_named_by_what_it_was_registered_as():
    _source.register_source_name("/var/cache/u/generated/abc/tilefish_generated/k.py", "tilefish_generated/k.py")
    assert _source.source_name("/var/cache/u/generated/abc/tilefish_generated/k.py", "tilefish_generated_k__abc") == "tilefish_generated/k.py"


def test_the_module_of_a_jit_function_is_where_its_python_function_is_defined():
    def f(x):
        return x

    assert _source.module_of(f) == __name__
    assert _source.module_of(fake_jit.jit(f)) == __name__
    assert _source.module_of(fake_jit.wrap(fake_jit.jit(f))) == __name__  # an autotuner holds the jit function as .fn
    assert _source.module_of(object()) is None


# -- the hook in the frontend ----------------------------------------------------------------------------------------------------------------------------


def test_the_frontend_names_files_by_module_path_only_in_the_thread_that_builds(fake, tmp_path, monkeypatch):
    kernel = import_from(checkout(tmp_path / "a"), monkeypatch)
    code_generator = sys.modules["triton.compiler.code_generator"]
    original = code_generator.get_jit_fn_file_line
    with _ir.host_independent_paths():
        hooked = code_generator.get_jit_fn_file_line
        assert hooked is not original and hooked.__wrapped__ is original
        assert hooked(kernel)[0] == "rb_pkg/kern.py"
        seen = []
        other = threading.Thread(target=lambda: seen.append(hooked(kernel)[0]))  # another thread's frontend (a Triton JIT launch of the program's own)
        other.start()
        other.join()
        assert seen == [str(tmp_path / "a" / "rb_pkg" / "kern.py")]
        with _ir.host_independent_paths():  # nested: installed once
            assert code_generator.get_jit_fn_file_line is hooked
        assert hooked(kernel)[0] == "rb_pkg/kern.py"
    assert hooked(kernel) == original(kernel)  # outside the block: what Triton says, line number included


def test_a_triton_without_the_entry_point_is_refused(fake, monkeypatch):
    monkeypatch.delattr(sys.modules["triton.compiler.code_generator"], "get_jit_fn_file_line")
    with pytest.raises(tilefish.TilefishBuildError, match="get_jit_fn_file_line"):
        with _ir.host_independent_paths():
            pass


def test_an_extern_built_from_two_checkouts_is_the_same_bytes(fake, tmp_path, monkeypatch):
    (spec,) = compile_specs(rb_extern)
    first = build(spec, arch=SM90, fn=import_from(checkout(tmp_path / "a"), monkeypatch), cache=False)
    second = build(spec, arch=SM90, fn=import_from(checkout(tmp_path / "b" / "a" / "much" / "deeper" / "checkout"), monkeypatch), cache=False)
    assert fake.count == 2  # two compiles
    assert debug_names(first.cubin) == debug_names(second.cubin) == "line:rb_pkg/kern.py"
    assert str(tmp_path).encode() not in first.cubin
    assert first.cubin == second.cubin and first.sha256 == second.sha256 and first.sha256_code == second.sha256_code


def test_two_exports_from_two_checkouts_and_two_cache_directories_are_the_same_files_and_digest(fake, tmp_path, monkeypatch):
    kernel = scale_kernel()
    out = []
    for i, root in enumerate((tmp_path / "a", tmp_path / "b" / "elsewhere")):
        import_from(checkout(root), monkeypatch)
        monkeypatch.setenv("TILEFISH_CACHE_DIR", str(root / "cache"))
        monkeypatch.setattr(runtime, "_modules", {})  # another process: the generated source is written again, to another directory
        out.append(runtime.export([kernel, rb_extern], tmp_path / f"out{i}", arch=SM90))
    assert fake.count == 4  # nothing came from a cache
    first, second = out
    assert first.digest == second.digest
    files = sorted(p.name for p in first.directory.iterdir())
    assert files == sorted(p.name for p in second.directory.iterdir()) and len(files) == 3
    for name in files:
        assert (first.directory / name).read_bytes() == (second.directory / name).read_bytes(), name


# -- a generated source: a content-addressed file ---------------------------------------------------------------------------------------------------------


def generated_path(low):
    key = hashlib.sha256(low.source.encode("utf-8")).hexdigest()
    return pathlib.Path(os.environ["TILEFISH_CACHE_DIR"]) / "generated" / key / low.filename


def test_a_generated_source_is_written_to_a_content_addressed_file_and_run_from_the_checked_text(fake):
    low = scale_kernel().lower()
    fn = runtime.load_lowered(low)
    path = generated_path(low)
    assert fn.__globals__["__file__"] == str(path) and path.read_bytes() == low.source.encode("utf-8")
    assert low.filename == "tilefish_generated/rb_scale.py"
    assert linecache.getlines(str(path)) == low.source.splitlines(keepends=True)  # what Triton reads with inspect.getsource
    assert not list(path.parent.glob("__pycache__")) and not list(path.parent.glob(".*.tmp"))  # no byte code cache, no temporary file left
    assert runtime.load_lowered(low) is fn  # cached by the sha256 of the source
    assert _source.source_name(str(path), _source.module_of(fn)) == "tilefish_generated/rb_scale.py"


def test_a_generated_kernel_is_the_same_bytes_from_another_cache_directory_and_as_lowered_write_writes_it(fake, tmp_path, monkeypatch):
    low = scale_kernel().lower()
    (spec,) = low.specs()
    first = build(spec, arch=SM90, fn=runtime.load_lowered(low), cache=False)
    monkeypatch.setenv("TILEFISH_CACHE_DIR", str(tmp_path / "another" / "cache"))
    monkeypatch.setattr(runtime, "_modules", {})  # another process
    fn = runtime.load_lowered(low)
    assert fn.__globals__["__file__"] == str(generated_path(low))
    second = build(spec, arch=SM90, fn=fn, cache=False)
    # the documented route of a toolchain build: LoweredKernel.write(directory), and an import of tilefish_generated.<name> from it
    low.write(tmp_path / "src")
    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    sys.modules.pop("tilefish_generated", None)
    sys.modules.pop("tilefish_generated.rb_scale", None)
    written = importlib.import_module("tilefish_generated.rb_scale").tf_rb_scale
    third = build(spec, arch=SM90, fn=written, cache=False)
    sys.modules.pop("tilefish_generated.rb_scale", None)
    sys.modules.pop("tilefish_generated", None)
    assert fake.count == 3
    assert debug_names(first.cubin) == "line:tilefish_generated/rb_scale.py"
    assert first.cubin == second.cubin == third.cubin


def test_a_file_that_does_not_hold_the_source_is_replaced(fake):
    low = scale_kernel().lower()
    path = generated_path(low)
    path.parent.mkdir(parents=True)
    path.write_text(low.source[: len(low.source) // 2])  # truncated (or tampered with)
    fn = runtime.load_lowered(low)
    assert fn.__globals__["__file__"] == str(path) and path.read_text() == low.source


def test_a_removed_file_is_put_back_for_a_function_that_is_cached(fake):
    low = scale_kernel().lower()
    fn = runtime.load_lowered(low)
    path = generated_path(low)
    path.unlink()
    assert runtime.load_lowered(low) is fn and path.read_text() == low.source


def test_a_symbolic_link_where_the_file_goes_is_not_followed(fake, tmp_path):
    low = scale_kernel().lower()
    path = generated_path(low)
    path.parent.mkdir(parents=True)
    target = tmp_path / "somewhere_else.py"
    target.write_text(low.source)
    path.symlink_to(target)
    with pytest.warns(TilefishCacheWarning, match="cannot keep the generated source"):
        fn = runtime.load_lowered(low)
    private = pathlib.Path(fn.__globals__["__file__"])
    assert private.is_relative_to(runtime._gen_root[0]) and private.read_text() == low.source and path.is_symlink()


def test_a_directory_that_cannot_be_written_falls_back_to_a_private_one_with_one_warning(fake, tmp_path, monkeypatch):
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("")
    monkeypatch.setenv("TILEFISH_CACHE_DIR", str(blocked))  # mkdir under a file fails, as under a read-only directory, for any user
    low, other = scale_kernel(2.0).lower(), scale_kernel(5.0).lower()
    (spec,) = low.specs()
    with pytest.warns(TilefishCacheWarning, match="cannot keep the generated source of 'rb_scale'") as record:
        fn = runtime.load_lowered(low)
        runtime.load_lowered(other)
    assert len([w for w in record if "generated source" in str(w.message)]) == 1  # one per directory
    path = pathlib.Path(fn.__globals__["__file__"])
    assert path.is_relative_to(runtime._gen_root[0]) and path.read_text() == low.source
    fallback = build(spec, arch=SM90, fn=fn, cache=False)
    monkeypatch.setenv("TILEFISH_CACHE_DIR", str(tmp_path / "writable"))
    monkeypatch.setattr(runtime, "_modules", {})
    assert build(spec, arch=SM90, fn=runtime.load_lowered(low), cache=False).cubin == fallback.cubin  # where the file is does not reach the binary


def test_a_switched_off_cache_keeps_generated_sources_in_the_private_directory_quietly(fake, monkeypatch):
    monkeypatch.setenv("TILEFISH_CACHE_DIR", "")
    low = scale_kernel().lower()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fn = runtime.load_lowered(low)
    assert pathlib.Path(fn.__globals__["__file__"]).is_relative_to(runtime._gen_root[0])


WRITER = r"""
import hashlib, pathlib, sys, time
import tilefish
from tilefish import runtime

M, D = tilefish.Dim("M"), tilefish.Dim("D", 64)


@tilefish.kernel(name="rb_scale")
def rb_scale(x: tilefish.Tensor[M, D, tilefish.f32]) -> tilefish.Tensor[M, D, tilefish.f32]:
    return x * 2.0


low = rb_scale.lower()
data = low.source.encode("utf-8")
key = hashlib.sha256(data).hexdigest()
start = pathlib.Path(sys.argv[1])
while not start.exists():
    time.sleep(0.001)
paths = {str(runtime._generated_file(low, key, data)) for _ in range(50)}
print(sorted(paths))
"""


def test_processes_that_write_the_same_generated_source_at_once_agree_on_one_file(tmp_path):
    cache = tmp_path / "shared-cache"
    env = dict(os.environ, TILEFISH_CACHE_DIR=str(cache), PYTHONPATH=os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]))
    start = tmp_path / "go"
    children = [
        subprocess.Popen([sys.executable, "-c", WRITER, str(start)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(6)
    ]
    start.write_text("")
    results = [child.communicate(timeout=120) for child in children]
    assert all(child.returncode == 0 for child in children), [err[-2000:] for _, err in results]
    paths = {out.strip() for out, _ in results}
    assert len(paths) == 1
    (path,) = json.loads(paths.pop().replace("'", '"'))
    path = pathlib.Path(path)
    assert path.is_relative_to(cache / "generated") and hashlib.sha256(path.read_bytes()).hexdigest() == path.parent.parent.name  # the source, whole
    assert not list(path.parent.glob(".*.tmp"))  # every temporary file was renamed into place or removed
