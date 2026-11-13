"""A module a kernel reaches that was edited after the process imported it is refused, before any digest is made of it.

The scenario (a notebook, a long-lived process): a module is imported, its file is edited without a reload, and ``export``
would compile the values the process holds while stamping the source digest of the file on disk, so a fresh process
replays the old binary without an error. These tests edit real files in a temporary package: a kernel, a ``cfg`` module,
and a helper that reads a constant of a helper below it (two levels away from the kernel's module).

No Triton and no GPU: ``fake_jit.FakeJit`` has what the checks read of a ``triton.JITFunction``. ``test_gpu_replay_safety.py``
and ``test_gpu_cache.py`` run the same scenarios through real kernels and real builds.
"""

import importlib
import linecache
import os
import sys

import pytest

from tilefish import TilefishBuildError, _source

MODULES = ("im_kernel", "im_cfg", "im_helper_a", "im_helper_b")

FILES = {
    "im_kernel": (
        "import im_cfg\n"
        "import im_helper_a\n"
        "from fake_jit import jit\n"
        "\n"
        "THRESH = 4\n"
        "\n"
        "\n"
        "@jit\n"
        "def kernel(x, SCALE):\n"
        "    y = im_helper_a.apply(x) * im_cfg.SCALE\n"
        "    return y * THRESH * SCALE\n"
    ),
    "im_cfg": 'SCALE = 2.0\nDT = "float32"\nTABLE = [1, 2]\n',
    "im_helper_a": (
        "import im_helper_b\n"
        "from fake_jit import jit\n"
        "\n"
        "\n"
        "@jit\n"
        "def apply(x):\n"
        "    return x + im_helper_b.OFFSET\n"
    ),
    "im_helper_b": "OFFSET = 0.0\n",
}


class Project:
    def __init__(self, directory):
        self.directory = directory
        for name, text in FILES.items():
            self.write(name, text)

    def path(self, name):
        return self.directory / f"{name}.py"

    def write(self, name, text):
        self.path(name).write_text(text)

    def edit(self, name, old, new, *, keep_mtime=False):
        text = self.path(name).read_text()
        assert old in text, (name, old)
        before = self.path(name).stat()
        self.write(name, text.replace(old, new, 1))
        if keep_mtime:  # the file looks untouched to anything that goes by its size and its modification time
            os.utime(self.path(name), ns=(before.st_atime_ns, before.st_mtime_ns))
            assert self.path(name).stat().st_mtime_ns == before.st_mtime_ns

    def load(self):
        """Import the project from its files as they are now (the way a new process would)."""
        for name in MODULES:
            sys.modules.pop(name, None)
        importlib.invalidate_caches()
        linecache.clearcache()
        return importlib.import_module("im_kernel")


@pytest.fixture
def project(tmp_path, monkeypatch):
    directory = tmp_path / "project"
    directory.mkdir()
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)  # an edit of one byte within a second must not meet a stale .pyc (see the gap below)
    monkeypatch.setattr(_source, "triton_version", lambda: "3.5.1")
    yield Project(directory)
    for name in MODULES:
        sys.modules.pop(name, None)


# -- the files ---------------------------------------------------------------------------------------------------------------


EDITS = [
    ("im_cfg", "SCALE = 2.0", "SCALE = 3.0"),  # the kernel reads it as im_cfg.SCALE; same size, not a line of any jit function
    ("im_helper_b", "OFFSET = 0.0", "OFFSET = 1.0"),  # two levels away from the kernel's module
    ("im_helper_a", "x + im_helper_b.OFFSET", "x - im_helper_b.OFFSET"),
    ("im_kernel", "THRESH = 4\n", "THRESH = 5\n"),  # a constant of the kernel's own module
    ("im_kernel", "THRESH = 4\n", "THRESH = 4  # a comment is an edit too\n"),
]


@pytest.mark.parametrize("name, old, new", EDITS, ids=[f"{n}:{o[:14]}" for n, o, _ in EDITS])
def test_a_file_edited_after_it_was_imported_is_refused_before_a_digest_is_made(project, name, old, new):
    kernel = project.load().kernel
    before = _source.source_digest(kernel)
    assert len(before) == 64
    project.edit(name, old, new)  # edited, not reloaded
    with pytest.raises(TilefishBuildError, match=rf"module '{name}' changed on disk since it was imported .*reload it") as e:
        _source.source_digest(kernel)
    assert isinstance(e.value, _source.StaleSourceError) and str(project.path(name)) in str(e.value) and "restart" in str(e.value)
    after = _source.source_digest(project.load().kernel)  # reloaded: the digest of what is on disk, and what the process holds
    assert after != before and len(after) == 64


def test_an_edit_that_keeps_the_size_and_the_modification_time_is_refused_too(project):
    """The check is of the content of the file as it was imported, not of its stat: any editor, rsync, ``touch -r``."""
    kernel = project.load().kernel
    project.edit("im_cfg", "SCALE = 2.0", "SCALE = 3.0", keep_mtime=True)
    with pytest.raises(_source.StaleSourceError, match="module 'im_cfg' changed on disk"):
        _source.source_digest(kernel)


def test_a_file_edited_and_restored_or_touched_is_not_a_change(project):
    kernel = project.load().kernel
    before = _source.source_digest(kernel)
    project.edit("im_cfg", "SCALE = 2.0", "SCALE = 3.0")
    project.edit("im_cfg", "SCALE = 3.0", "SCALE = 2.0")  # back to what the process imported
    assert _source.source_digest(kernel) == before
    later = project.path("im_cfg").stat().st_mtime_ns + 5_000_000_000
    os.utime(project.path("im_cfg"), ns=(later, later))  # a save without a change: the content is the one imported
    assert _source.source_digest(kernel) == before


def test_an_untouched_project_is_never_refused(project):
    kernel = project.load().kernel
    digests = {_source.source_digest(kernel) for _ in range(5)}
    assert len(digests) == 1
    _source.check_current(_source.reachable(kernel))


def test_the_edit_is_refused_by_every_path_that_computes_a_digest(project):
    kernel = project.load().kernel
    found = _source.reachable(kernel)
    project.edit("im_helper_b", "OFFSET = 0.0", "OFFSET = 1.0")
    with pytest.raises(_source.StaleSourceError):
        _source.digest_of(found)
    with pytest.raises(_source.StaleSourceError):
        _source.check_current(found)
    with pytest.raises(_source.StaleSourceError):
        _source.source_digest(kernel)


def test_reloading_the_module_that_changed_is_what_clears_it(project):
    kernel = project.load().kernel
    project.edit("im_cfg", "SCALE = 2.0", "SCALE = 3.0")
    with pytest.raises(_source.StaleSourceError):
        _source.source_digest(kernel)
    importlib.reload(sys.modules["im_cfg"])  # the process now holds the file
    digest = _source.source_digest(kernel)
    assert digest == _source.source_digest(project.load().kernel)  # what a fresh process would compute


def test_a_module_the_kernel_does_not_reach_is_not_looked_at(project):
    kernel = project.load().kernel
    project.write("im_unrelated", "X = 1\n")
    sys.path.insert(0, str(project.directory))
    try:
        import im_unrelated  # noqa: F401
    finally:
        sys.path.remove(str(project.directory))
        sys.modules.pop("im_unrelated", None)
    project.write("im_unrelated", "X = 2\n")
    assert _source.source_digest(kernel) is not None


def test_a_module_imported_with_spec_from_file_location_is_watched(project):
    import importlib.util

    project.write("im_loose", "VALUE = 1\n")
    spec = importlib.util.spec_from_file_location("im_loose", project.path("im_loose"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = _source._watch.imported[os.path.abspath(project.path("im_loose"))]
    assert record.values == {"VALUE": 1}
    project.write("im_loose", "VALUE = 2\n")
    found = _source.Sources({"im_loose": str(project.path("im_loose"))}, [])
    with pytest.raises(_source.StaleSourceError, match="module 'im_loose' changed on disk"):
        _source.check_current(found)


def test_the_standard_library_and_tilefish_are_not_recorded(project):
    import json.tool  # noqa: F401

    for name in ("json", "tilefish", "tilefish._source"):
        path = os.path.abspath(sys.modules[name].__file__)
        assert path not in _source._watch.imported, name


def test_the_watch_does_not_break_an_import_that_fails(project):
    project.write("im_broken", "def f(:\n")
    sys.path.insert(0, str(project.directory))
    try:
        with pytest.raises(SyntaxError):
            import im_broken  # noqa: F401
        project.write("im_broken", "raise RuntimeError('boom')\n")
        importlib.invalidate_caches()
        with pytest.raises(RuntimeError, match="boom"):
            import im_broken  # noqa: F401, F811
    finally:
        sys.path.remove(str(project.directory))
        sys.modules.pop("im_broken", None)


def test_the_watch_installs_once_and_can_be_removed(project):
    watch = _source._watch
    assert watch.installed and importlib.machinery.SourceFileLoader.exec_module._tilefish_watch is watch
    wrapper = importlib.machinery.SourceFileLoader.exec_module
    watch.install()  # a second call changes nothing
    assert importlib.machinery.SourceFileLoader.exec_module is wrapper
    watch.uninstall()
    try:
        assert not watch.installed and not hasattr(importlib.machinery.SourceFileLoader.exec_module, "_tilefish_watch")
    finally:
        watch.install()
    assert importlib.machinery.SourceFileLoader.exec_module._tilefish_watch is watch


def test_importing_tilefish_installs_the_watch_in_a_new_process():
    import subprocess

    code = (
        "import importlib.machinery as m\n"
        "assert not hasattr(m.SourceFileLoader.exec_module, '_tilefish_watch')\n"
        "import tilefish\n"
        "assert hasattr(m.SourceFileLoader.exec_module, '_tilefish_watch')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# -- what was imported before tilefish -------------------------------------------------------------------------------------


@pytest.fixture
def imported_before_tilefish(project):
    """The project imported while the watch is off, then the watch installed: the state of a module a program imported before ``import tilefish``."""
    watch = _source._watch
    watch.uninstall()
    try:
        kernel = project.load().kernel
    finally:
        watch.install()
    assert all(os.path.abspath(project.path(n)) not in watch.imported for n in MODULES)
    assert all(os.path.abspath(project.path(n)) in watch.before for n in MODULES)
    return kernel


def test_a_module_imported_before_tilefish_is_refused_when_its_size_or_time_changed_since(imported_before_tilefish, project):
    kernel = imported_before_tilefish
    digest = _source.source_digest(kernel)  # unchanged: fine
    assert len(digest) == 64
    project.edit("im_helper_b", "OFFSET = 0.0", "OFFSET = 1.0")  # same size, a later modification time
    later = project.path("im_helper_b").stat().st_mtime_ns + 1_000_000_000
    os.utime(project.path("im_helper_b"), ns=(later, later))
    with pytest.raises(_source.StaleSourceError, match="module 'im_helper_b' changed on disk since it was imported .*imported before tilefish"):
        _source.source_digest(kernel)


def test_residual_gap_a_module_edited_between_its_import_and_import_tilefish_is_not_seen(project):
    """Documented, not wanted: nothing says what the process imported, only what the file is when the watch starts."""
    watch = _source._watch
    watch.uninstall()
    try:
        kernel = project.load().kernel
        project.edit("im_cfg", "SCALE = 2.0", "SCALE = 3.0")  # the process holds 2.0
    finally:
        watch.install()
    assert _source.source_digest(kernel) is not None  # and a build would stamp the digest of 3.0 on a binary of 2.0


def test_residual_gap_a_stale_pyc_is_what_python_ran_and_the_file_is_what_the_watch_recorded(project, monkeypatch):
    """CPython trusts a ``.pyc`` whose header has the source's size and whole-second mtime, so an edit that keeps both is not
    reloaded; the watch records the file, not the bytecode that ran, and cannot tell. (``checked-hash`` pycs are exact.)"""
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    kernel = project.load().kernel  # writes the .pyc of im_cfg from SCALE = 2.0
    project.edit("im_cfg", "SCALE = 2.0", "SCALE = 3.0", keep_mtime=True)
    importlib.reload(sys.modules["im_cfg"])
    assert sys.modules["im_cfg"].SCALE == 2.0  # the process still holds 2.0 ...
    assert "SCALE = 3.0" in project.path("im_cfg").read_text()  # ... and the file says 3.0
    assert _source.source_digest(kernel) is not None  # not refused: the gap


def test_residual_gap_an_edit_that_keeps_size_and_time_of_a_module_imported_before_tilefish_is_not_seen(imported_before_tilefish, project):
    project.edit("im_cfg", "SCALE = 2.0", "SCALE = 3.0", keep_mtime=True)
    assert _source.source_digest(imported_before_tilefish) is not None


def test_reloading_makes_a_module_imported_before_tilefish_exact(imported_before_tilefish, project):
    kernel = imported_before_tilefish
    project.edit("im_cfg", "SCALE = 2.0", "SCALE = 3.0")
    with pytest.raises(_source.StaleSourceError):
        _source.source_digest(kernel)
    importlib.reload(sys.modules["im_cfg"])
    assert _source.source_digest(kernel) is not None
    project.edit("im_cfg", "SCALE = 3.0", "SCALE = 4.0", keep_mtime=True)  # now recorded: the content is what counts
    with pytest.raises(_source.StaleSourceError):
        _source.source_digest(kernel)


# -- the values --------------------------------------------------------------------------------------------------------------


def test_a_module_attribute_a_kernel_reads_must_not_be_assigned_at_run_time(project):
    kernel = project.load().kernel
    before = _source.source_digest(kernel)
    sys.modules["im_cfg"].SCALE = 5.0
    with pytest.raises(TilefishBuildError, match=r"module attribute 'im_cfg.SCALE', which the Triton function 'kernel' reads, was assigned after 'im_cfg' was imported \(it was 2.0, it is 5.0\)"):
        _source.source_digest(kernel)
    sys.modules["im_cfg"].SCALE = 2.0  # put back: the same value
    assert _source.source_digest(kernel) == before
    sys.modules["im_cfg"].SCALE = 2.0 + 0  # an equal float is not a change
    assert _source.source_digest(kernel) == before


@pytest.mark.parametrize(
    "module, name, value",
    [
        ("im_kernel", "THRESH", 5),  # a name the kernel reads from its own globals
        ("im_helper_b", "OFFSET", 1.0),  # two levels away: read by `apply` as im_helper_b.OFFSET
    ],
)
def test_a_global_read_by_name_and_a_constant_two_levels_away_are_checked_too(project, module, name, value):
    kernel = project.load().kernel
    setattr(sys.modules[module], name, value)
    with pytest.raises(_source.StaleSourceError, match=rf"'{module}.{name}'"):
        _source.source_digest(kernel)


def test_a_deleted_attribute_and_a_changed_type_are_changes(project):
    kernel = project.load().kernel
    del sys.modules["im_cfg"].SCALE
    with pytest.raises(_source.StaleSourceError, match="it is deleted"):
        _source.source_digest(kernel)
    sys.modules["im_cfg"].SCALE = 2  # an int where there was a float
    with pytest.raises(_source.StaleSourceError, match="it was 2.0, it is 2"):
        _source.source_digest(kernel)


def test_what_the_kernel_does_not_read_may_change(project):
    kernel = project.load().kernel
    before = _source.source_digest(kernel)
    sys.modules["im_cfg"].DT = "bfloat16"  # a scalar of a reachable module that no kernel function mentions
    sys.modules["im_cfg"].NEW = 1
    sys.modules["im_cfg"].TABLE.append(3)  # a container changed in place is not seen (documented)
    assert _source.source_digest(kernel) == before


def test_a_parameter_of_the_kernel_that_has_the_name_of_a_module_scalar_is_not_a_read(project):
    """``SCALE`` is a parameter of ``kernel`` and also a global of ``im_cfg``: the kernel's own ``SCALE`` is local."""
    kernel = project.load().kernel
    sys.modules["im_kernel"].SCALE = 99  # the kernel module has no such global at import; assigning one is not a change of a recorded value
    assert _source.source_digest(kernel) is not None


def test_a_module_the_watch_did_not_see_has_no_values_to_compare(imported_before_tilefish, project):
    sys.modules["im_cfg"].SCALE = 5.0  # no snapshot of a module that was imported before the watch: a documented gap
    assert _source.source_digest(imported_before_tilefish) is not None


def test_the_values_of_a_reloaded_module_are_the_new_baseline(project):
    kernel = project.load().kernel
    project.edit("im_cfg", "SCALE = 2.0", "SCALE = 3.0")
    importlib.reload(sys.modules["im_cfg"])
    assert sys.modules["im_cfg"].SCALE == 3.0 and _source.source_digest(kernel) is not None
    sys.modules["im_cfg"].SCALE = 2.0
    with pytest.raises(_source.StaleSourceError, match="it was 3.0, it is 2.0"):
        _source.source_digest(kernel)
