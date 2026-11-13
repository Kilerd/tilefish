"""``TILEFISH_IMPORT_WATCH=0``: tilefish leaves the import system alone and falls back to size and modification time."""

import hashlib
import importlib
import importlib.util
import os
import subprocess
import sys
import textwrap
import types

import pytest

from tilefish import _source

SCRIPT = textwrap.dedent(
    """
    import importlib, importlib.machinery, os, sys, time
    sys.path.insert(0, sys.argv[1])
    import tilefish
    from tilefish import _source

    wrapped = hasattr(importlib.machinery.SourceFileLoader.exec_module, "_tilefish_watch")
    print("wrapped", wrapped, "lazy", _source._watch.lazy)

    path = os.path.join(sys.argv[1], "late_mod.py")
    with open(path, "w") as f:
        f.write("X = 1\\n")
    importlib.invalidate_caches()
    import late_mod

    def check():
        with open(path, "rb") as f:
            data = f.read()
            st = os.fstat(f.fileno())
        import hashlib
        _source._watch.check_file("late_mod", path, hashlib.sha256(data).hexdigest(), st)

    check()  # first read by a build: recorded (lazy) or compared with the import record (watched)
    with open(path, "w") as f:
        f.write("X = 22\\n")  # another size, so the fallback sees it even within one mtime tick
    try:
        check()
        print("edit", "passed")
    except _source.StaleSourceError:
        print("edit", "refused")
    """
)


def _run(tmp_path, env_value):
    env = dict(os.environ)
    env.pop("TILEFISH_IMPORT_WATCH", None)
    if env_value is not None:
        env["TILEFISH_IMPORT_WATCH"] = env_value
    script = tmp_path / "probe.py"
    script.write_text(SCRIPT)
    out = subprocess.run(
        [sys.executable, str(script), str(tmp_path)], env=env, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    return out[0], out[1]


def test_default_wraps_the_loader_and_refuses_an_edit(tmp_path):
    first, edit = _run(tmp_path, None)
    assert first == "wrapped True lazy False"
    assert edit == "edit refused"


def test_opt_out_leaves_the_loader_alone_and_still_refuses_an_edit_after_the_first_read(tmp_path):
    first, edit = _run(tmp_path, "0")
    assert first == "wrapped False lazy True"
    assert edit == "edit refused"


def test_any_other_value_keeps_the_watch(tmp_path):
    first, _ = _run(tmp_path, "1")
    assert first == "wrapped True lazy False"


# -- the values that switch it off ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "false", "False", "FALSE", "no", "No", "NO", "off", "Off", "OFF", " off ", "\tFalse\n"])
def test_the_watch_is_switched_off_by_zero_false_no_or_off_in_any_case(value, monkeypatch):
    watch = _source._Watch()
    monkeypatch.setattr(_source, "_watch", watch)
    monkeypatch.setenv("TILEFISH_IMPORT_WATCH", value)
    try:
        _source.install_import_watch()
        assert watch.lazy and not watch.installed
    finally:
        watch.uninstall()


@pytest.mark.parametrize("value", ["", "1", "true", "yes", "on", "2", "00", "offf", "of", "nope", "enabled"])
def test_any_other_value_keeps_the_watch_on(value, monkeypatch):
    watch = _source._Watch()
    monkeypatch.setattr(_source, "_watch", watch)
    monkeypatch.setenv("TILEFISH_IMPORT_WATCH", value)
    try:
        _source.install_import_watch()
        assert watch.installed and not watch.lazy
    finally:
        watch.uninstall()


# -- a module that is loaded lazily is never loaded by looking at it ------------------------------------------------------------------------

LAZY_BODY = 'import os\n\nopen(os.environ["LZ_MARKER"], "a").write("ran\\n")\nSCALE = 2.0\n'

LAZY_SCRIPT = textwrap.dedent(
    """
    import importlib.util, os, sys
    sys.path.insert(0, sys.argv[1])
    path = os.path.join(sys.argv[1], "lz_mod.py")
    spec = importlib.util.spec_from_file_location("lz_mod", path)
    spec.loader = importlib.util.LazyLoader(spec.loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lz_mod"] = module
    spec.loader.exec_module(module)  # lazy: the body has not run
    marker = os.environ["LZ_MARKER"]
    print("before", os.path.exists(marker))
    import tilefish
    from tilefish import _source
    print("imported", os.path.exists(marker), "lazy", _source._watch.lazy)
    module.SCALE  # a real use loads it
    print("used", os.path.exists(marker))
    """
)


def _lazy_run(tmp_path, env_value):
    (tmp_path / "lz_mod.py").write_text(LAZY_BODY)
    env = dict(os.environ)
    env.pop("TILEFISH_IMPORT_WATCH", None)
    env["LZ_MARKER"] = str(tmp_path / "marker")
    if env_value is not None:
        env["TILEFISH_IMPORT_WATCH"] = env_value
    script = tmp_path / "lazy_probe.py"
    script.write_text(LAZY_SCRIPT)
    return subprocess.run([sys.executable, str(script), str(tmp_path)], env=env, capture_output=True, text=True, check=True).stdout.split("\n")[:3]


@pytest.mark.parametrize("env_value, lazy", [(None, "False"), ("0", "True"), ("off", "True"), ("FALSE", "True"), ("1", "False")])
def test_importing_tilefish_does_not_run_the_body_of_a_lazy_module(tmp_path, env_value, lazy):
    """``import tilefish`` asked every module in ``sys.modules`` for ``__file__``, which executes a pending ``LazyLoader`` body."""
    before, imported, used = _lazy_run(tmp_path, env_value)
    assert before == "before False"
    assert imported == f"imported False lazy {lazy}"  # the body did not run: the side effect (the marker) is not there
    assert used == "used True"  # and it runs when the module is used, once
    assert (tmp_path / "marker").read_text() == "ran\n"


def _lazy_module(directory):
    (directory / "lz_mod.py").write_text(LAZY_BODY)
    spec = importlib.util.spec_from_file_location("lz_mod", directory / "lz_mod.py")
    spec.loader = importlib.util.LazyLoader(spec.loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lz_mod"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def lazy(tmp_path, monkeypatch):
    monkeypatch.setenv("LZ_MARKER", str(tmp_path / "marker"))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    module = _lazy_module(tmp_path)
    assert not (tmp_path / "marker").exists()
    yield module, tmp_path / "marker"
    sys.modules.pop("lz_mod", None)


@pytest.mark.parametrize("off", [False, True])
def test_neither_the_baseline_nor_a_digest_loads_a_lazy_module(lazy, off, tmp_path, monkeypatch):
    module, marker = lazy
    watch = _source._Watch()
    monkeypatch.setattr(_source, "_watch", watch)
    monkeypatch.setenv("TILEFISH_IMPORT_WATCH", "0" if off else "1")
    monkeypatch.setattr(_source, "triton_version", lambda: "3.5.1")
    try:
        _source.install_import_watch()
        watch.record_baseline()
        assert not marker.exists()
        assert os.path.abspath(str(tmp_path / "lz_mod.py")) in watch.before  # it is looked at by its file, which is all it has
        # a kernel that holds the lazy module, reads an attribute of it, and calls a jit function of it
        kernel_module = types.ModuleType("lz_kernel_module")
        (tmp_path / "lz_kernel_module.py").write_text("def kernel(x):\n    return lz_mod.SCALE * x\n")
        kernel_module.__dict__.update({"__file__": str(tmp_path / "lz_kernel_module.py"), "lz_mod": module})
        exec(compile((tmp_path / "lz_kernel_module.py").read_text(), str(tmp_path / "lz_kernel_module.py"), "exec"), kernel_module.__dict__)
        import fake_jit

        jit = fake_jit.jit(kernel_module.kernel)
        found = _source.reachable(jit)
        assert not marker.exists() and "lz_mod" in found.files  # reached by its file, which is what it will run from
        assert _source.values_read(found, strict=True) == {}  # nothing of it is read (it has not run): not loaded to find out
        assert _source.source_digest(jit) is not None
        _source.check_current(found)
        assert not marker.exists()
    finally:
        watch.uninstall()
    module.SCALE  # a use loads it, once
    assert marker.read_text() == "ran\n"


# -- reloading a module in the lazy mode ------------------------------------------------------------------------------------------------------


def _check(watch, name, path):
    with open(path, "rb") as f:
        data = f.read()
        st = os.fstat(f.fileno())
    watch.check_file(name, path, hashlib.sha256(data).hexdigest(), st)


def test_a_module_reloaded_after_an_edit_is_recorded_again_in_the_lazy_mode_and_not_refused_forever(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    path = tmp_path / "rl_mod.py"
    path.write_text("X = 1\n")
    importlib.invalidate_caches()
    module = importlib.import_module("rl_mod")
    try:
        watch = _source._Watch(lazy=True)
        watch.record_baseline()
        _check(watch, "rl_mod", str(path))
        path.write_text("X = 22\n")  # edited, not reloaded: the process holds the old one
        with pytest.raises(_source.StaleSourceError, match="changed on disk since it was imported"):
            _check(watch, "rl_mod", str(path))
        importlib.reload(module)  # now it holds what the file has
        assert module.X == 22
        _check(watch, "rl_mod", str(path))  # recorded again, not "changed on disk" for ever
        _check(watch, "rl_mod", str(path))
        path.write_text("X = 333\n")  # and an edit after the reload is an edit again
        with pytest.raises(_source.StaleSourceError, match="changed on disk since it was imported"):
            _check(watch, "rl_mod", str(path))
        importlib.reload(module)
        assert module.X == 333
        _check(watch, "rl_mod", str(path))
    finally:
        sys.modules.pop("rl_mod", None)


def test_a_module_imported_again_under_the_same_name_is_recorded_again_too(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    path = tmp_path / "rl_mod2.py"
    path.write_text("X = 1\n")
    importlib.invalidate_caches()
    importlib.import_module("rl_mod2")
    try:
        watch = _source._Watch(lazy=True)
        watch.record_baseline()
        path.write_text("X = 22\n")
        with pytest.raises(_source.StaleSourceError):
            _check(watch, "rl_mod2", str(path))
        sys.modules.pop("rl_mod2")
        importlib.invalidate_caches()
        assert importlib.import_module("rl_mod2").X == 22  # a new module object, a new spec
        _check(watch, "rl_mod2", str(path))
    finally:
        sys.modules.pop("rl_mod2", None)


def test_the_watched_mode_records_a_reload_by_its_loader_as_before(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    path = tmp_path / "rl_mod3.py"
    path.write_text("X = 1\n")
    importlib.invalidate_caches()
    watch = _source._Watch()
    monkeypatch.setattr(_source, "_watch", watch)
    try:
        watch.install()
        module = importlib.import_module("rl_mod3")
        _check(watch, "rl_mod3", str(path))
        path.write_text("X = 22\n")
        with pytest.raises(_source.StaleSourceError):
            _check(watch, "rl_mod3", str(path))
        importlib.reload(module)
        _check(watch, "rl_mod3", str(path))
    finally:
        watch.uninstall()
        sys.modules.pop("rl_mod3", None)
