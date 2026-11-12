"""The source digest of a hand-written kernel: which files it covers, what it excludes, when it cannot be computed.

No Triton and no GPU: ``fake_jit.FakeJit`` has what the digest reads of a ``triton.JITFunction`` (its function, its globals
and the lines it was imported from), and the Triton version is patched. The GPU tests in ``test_gpu_replay_safety.py`` run the
same scenarios through real kernels, real builds and a real replay.

The reason it exists: Triton's ``cache_key`` does not see a ``tl.constexpr`` read as a module attribute (``cfg.SCALE``), a
dtype alias (``DT = tl.float32``) or a constant in a helper module, so none of them made a stale binary look stale.
"""

import importlib
import linecache
import shutil
import sys

import pytest

from tilefish import TilefishBuildError, _source

MODULES = ("sd_kernel", "sd_cfg", "sd_helper_a", "sd_helper_b", "sd_other")

FILES = {
    "sd_kernel": (
        "import sd_cfg\n"
        "import sd_helper_a\n"
        "from fake_jit import jit\n"
        "\n"
        "\n"
        "@jit\n"
        "def kernel(x):\n"
        "    return sd_helper_a.apply(x) * sd_cfg.SCALE\n"
    ),
    "sd_cfg": 'SCALE = 2.0\nDT = "float32"\n\n\nclass Config:\n    pass\n',
    "sd_helper_a": (
        "import sd_helper_b\n"
        "from fake_jit import jit\n"
        "\n"
        "\n"
        "@jit\n"
        "def apply(x):\n"
        "    return x + sd_helper_b.OFFSET\n"
    ),
    "sd_helper_b": "OFFSET = 0.0\n",
}


class Project:
    def __init__(self, directory):
        self.directory = directory
        for name, text in FILES.items():
            self.write(name, text)

    def write(self, name, text):
        (self.directory / f"{name}.py").write_text(text)

    def edit(self, name, old, new):
        text = (self.directory / f"{name}.py").read_text()
        assert old in text, (name, old)
        self.write(name, text.replace(old, new, 1))

    def load(self):
        """Import the project from its files as they are now (the way a new process would)."""
        for name in MODULES:
            sys.modules.pop(name, None)
        importlib.invalidate_caches()
        linecache.clearcache()
        return importlib.import_module("sd_kernel")


@pytest.fixture
def project(tmp_path, monkeypatch):
    directory = tmp_path / "project"
    directory.mkdir()
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)  # an edit of one byte within a second must not meet a stale .pyc
    monkeypatch.setattr(_source, "triton_version", lambda: "3.5.1")
    yield Project(directory)
    for name in MODULES:
        sys.modules.pop(name, None)


def digest_of(project):
    return _source.source_digest(project.load().kernel)


# -- what is covered ----------------------------------------------------------------------------------------------------


def test_the_files_a_kernel_reaches_are_its_module_and_every_user_module_below_it(project):
    found = _source.reachable(project.load().kernel)
    assert set(found.files) == {"sd_kernel", "sd_cfg", "sd_helper_a", "sd_helper_b", "fake_jit"}
    assert found.files["sd_helper_b"] == str(project.directory / "sd_helper_b.py")
    assert [f.__name__ for f in found.functions] == ["kernel", "apply"]  # the jit functions among them: what a build checks


@pytest.mark.parametrize(
    "name, old, new",
    [
        ("sd_kernel", "* sd_cfg.SCALE", "* sd_cfg.SCALE  # a comment is an edit too"),
        ("sd_cfg", "SCALE = 2.0", "SCALE = 3.0"),  # a constexpr read as a module attribute: cfg.SCALE
        ("sd_cfg", 'DT = "float32"', 'DT = "bfloat16"'),  # a dtype alias
        ("sd_helper_a", "x + sd_helper_b.OFFSET", "x - sd_helper_b.OFFSET"),
        ("sd_helper_b", "OFFSET = 0.0", "OFFSET = 1.0"),  # two levels away from the kernel's module
    ],
)
def test_an_edit_in_any_reachable_file_changes_the_digest(project, name, old, new):
    before = digest_of(project)
    assert before == digest_of(project)  # a reload is not a change
    project.edit(name, old, new)
    after = digest_of(project)
    assert after != before and len(after) == 64
    project.edit(name, new, old)
    assert digest_of(project) == before  # and the edit undone is the digest it was


def test_the_digest_is_of_the_files_not_of_where_they_are(project, tmp_path, monkeypatch):
    before = digest_of(project)
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(project.directory, elsewhere)
    monkeypatch.syspath_prepend(str(elsewhere))  # first on the path: this is where the modules come from now
    kernel = project.load().kernel
    assert _source.reachable(kernel).files["sd_cfg"] == str(elsewhere / "sd_cfg.py")
    assert _source.source_digest(kernel) == before


def test_the_triton_version_is_part_of_the_digest(project, monkeypatch):
    before = digest_of(project)
    monkeypatch.setattr(_source, "triton_version", lambda: "3.6.0")
    assert digest_of(project) != before


# -- what is followed ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line, reached",
    [
        ("from sd_other import Thing\n", True),  # a class: the module it was defined in
        ("from sd_other import Thing as _T\nINSTANCE = _T()\ndel _T\n", True),  # an instance: the module of its class
        ("from sd_other import helper\n", True),  # a plain function: its globals
        ("import sd_other as o\n", True),  # a module under another name
        ("from sd_other import Opaque as _O\nOPAQUE = _O()\ndel _O\n", True),  # an instance that answers no attribute: its class says
        ("from sd_other import VALUE\n", False),  # an int: the value was read at import, nothing says where it came from
    ],
)
def test_a_module_is_reached_through_a_class_an_instance_a_function_or_a_module(project, line, reached):
    project.write(
        "sd_other",
        "class Thing:\n    pass\n\n\nclass Opaque:\n    def __getattribute__(self, name):\n        raise RuntimeError(name)\n\n\n"
        "def helper():\n    pass\n\n\nVALUE = 1\n",
    )
    project.write("sd_kernel", FILES["sd_kernel"].replace("import sd_cfg\n", "import sd_cfg\n" + line, 1))
    assert ("sd_other" in _source.reachable(project.load().kernel).files) is reached


def test_a_function_wrapped_by_an_autotuner_is_followed_through_fn(project):
    """``triton.autotune`` and ``triton.heuristics`` hold the jit function as ``.fn``; the wrapper's own module is Triton's."""
    project.write("sd_other", "def helper():\n    pass\n")
    text = FILES["sd_kernel"].replace(
        "from fake_jit import jit\n", "from fake_jit import jit, wrap\nimport sd_other as _o\nTUNED = wrap(_o.helper)\ndel _o\n", 1
    )
    project.write("sd_kernel", text)
    kernel = project.load().kernel
    assert "sd_other" in _source.reachable(kernel).files
    assert _source.jit_function(kernel) is kernel
    wrapper = type("Wrapper", (), {})()
    wrapper.fn = kernel
    assert _source.jit_function(wrapper) is kernel  # an autotuned kernel is read through to its jit function


def test_a_container_of_functions_is_not_opened_and_a_function_level_import_is_not_seen(project):
    """The documented boundary: only the top level of a namespace is followed."""
    project.write("sd_other", "def helper():\n    pass\n")
    project.write(
        "sd_kernel",
        FILES["sd_kernel"].replace("import sd_cfg\n", "import sd_cfg\nimport sd_other as _o\nTABLE = [_o.helper]\ndel _o\n", 1),
    )
    assert "sd_other" not in _source.reachable(project.load().kernel).files


def test_standard_library_triton_torch_and_tilefish_modules_are_not_user_modules():
    for name in ("os", "os.path", "json", "typing", "math", "functools", "collections.abc", "builtins", "__future__", "triton", "triton.language.core", "torch", "torch.nn.functional", "tilefish", "tilefish.aot"):
        assert not _source.is_user_module(name), name
    for name in ("numpy", "my_kernels", "sd_cfg", "tilefish_extras", "torchvision", "triton_kernels", "__main__", "helpers.cfg"):
        assert _source.is_user_module(name), name


def test_the_standard_library_is_not_read(project):
    project.write("sd_kernel", "import os\nimport json\nimport sd_cfg\n" + FILES["sd_kernel"].replace("import sd_cfg\n", "", 1))
    found = _source.reachable(project.load().kernel)
    assert "os" not in found.files and "json" not in found.files and "sd_cfg" in found.files


def test_a_module_without_a_python_file_contributes_nothing_and_does_not_stop_the_walk(project):
    import types

    bare = types.ModuleType("sd_other")  # no __file__: a built-in or an extension module looks like this
    sys.modules["sd_other"] = bare
    project.write("sd_kernel", FILES["sd_kernel"].replace("import sd_cfg\n", "import sd_cfg\nimport sd_other\n", 1))
    for name in ("sd_kernel", "sd_cfg", "sd_helper_a", "sd_helper_b"):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    kernel = importlib.import_module("sd_kernel").kernel
    assert set(_source.reachable(kernel).files) == {"sd_kernel", "sd_cfg", "sd_helper_a", "sd_helper_b", "fake_jit"}
    assert _source.source_digest(kernel) is not None


@pytest.mark.parametrize("suffix", [".so", ".pyc", ".pyd", ".txt", ""])
def test_only_python_source_files_are_read(project, suffix):
    """An extension module or a bytecode-only file has no text to hash: it contributes nothing (and is not read as if it had)."""
    import types

    binary = project.directory / f"sd_other_binary{suffix}"
    binary.write_bytes(b"\x7fELF not text")
    module = types.ModuleType("sd_other")
    module.__file__ = str(binary)
    sys.modules["sd_other"] = module
    project.write("sd_kernel", FILES["sd_kernel"].replace("import sd_cfg\n", "import sd_cfg\nimport sd_other\n", 1))
    for name in ("sd_kernel", "sd_cfg", "sd_helper_a", "sd_helper_b"):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    assert "sd_other" not in _source.reachable(importlib.import_module("sd_kernel").kernel).files


def test_each_file_counts_under_its_module_name_and_in_any_order(tmp_path, monkeypatch):
    monkeypatch.setattr(_source, "triton_version", lambda: "3.5.1")
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("A = 1\n")
    b.write_text("B = 2\n")
    same_text_other_name = _source.Sources({"x": str(a), "y": str(b)}, [])
    assert _source.digest_of(same_text_other_name) != _source.digest_of(_source.Sources({"p": str(a), "q": str(b)}, []))
    assert _source.digest_of(same_text_other_name) == _source.digest_of(_source.Sources({"y": str(b), "x": str(a)}, []))  # dict order is not part of it
    swapped = _source.Sources({"x": str(b), "y": str(a)}, [])  # the same names over the same texts, swapped
    assert _source.digest_of(swapped) != _source.digest_of(same_text_other_name)
    a.write_text("A = 3\n")
    assert _source.digest_of(same_text_other_name) != _source.digest_of(swapped)
    assert len(_source.digest_of(same_text_other_name)) == 64
    assert _source.digest_of(None) is None


def test_modules_that_import_each_other_are_read_once(project):
    project.write("sd_helper_b", "import sd_helper_a\nOFFSET = 0.0\n")
    found = _source.reachable(project.load().kernel)
    assert sorted(found.files) == ["fake_jit", "sd_cfg", "sd_helper_a", "sd_helper_b", "sd_kernel"]


def test_an_object_whose_attributes_raise_is_looked_past(project):
    project.write("sd_cfg", FILES["sd_cfg"] + "\n\nclass Hostile:\n    def __getattr__(self, name):\n        raise RuntimeError(name)\n\n\nHOSTILE = Hostile()\n")
    assert _source.source_digest(project.load().kernel) is not None


# -- when there is no digest -----------------------------------------------------------------------------------------------------


def test_there_is_no_digest_without_triton(project, monkeypatch):
    kernel = project.load().kernel
    assert _source.source_digest(kernel) is not None
    monkeypatch.setattr(_source, "triton_version", lambda: None)
    assert _source.source_digest(kernel) is None


def test_there_is_no_digest_for_a_kernel_whose_module_is_not_a_file(project):
    namespace = {"__name__": "sd_inline"}
    exec(compile("def kernel(x):\n    return x\n", "<string>", "exec"), namespace)
    fake = type("Inline", (), {"fn": namespace["kernel"], "raw_src": ["def kernel(x):\n"], "starting_line_number": 1, "__globals__": namespace})()
    assert _source.source_digest(fake) is None and _source.function_file(fake) is None
    assert _source.source_digest(object()) is None and _source.reachable(None) is None


def test_there_is_no_digest_when_a_file_it_reached_is_gone(project):
    found = _source.reachable(project.load().kernel)
    (project.directory / "sd_helper_b.py").unlink()
    assert _source.digest_of(found) is None
    with pytest.raises(OSError):
        _source.file_sha256(found.files["sd_helper_b"])


# -- what the process imported against what is on disk ---------------------------------------------------------------------------


def test_the_lines_a_function_was_imported_from_must_be_the_ones_in_its_file(project):
    found = _source.reachable(project.load().kernel)
    _source.check_imported_lines(found.functions)  # unchanged: fine
    project.edit("sd_helper_a", "x + sd_helper_b.OFFSET", "x - sd_helper_b.OFFSET")  # edited after it was imported
    with pytest.raises(TilefishBuildError, match=r"'apply' was imported from an older version of '.*sd_helper_a.py': its lines 5-7 are not what the file has now"):
        _source.check_imported_lines(found.functions)
    with pytest.raises(TilefishBuildError, match="Restart the process, or reload the module"):
        _source.check_imported_lines(found.functions)


def test_an_edit_above_a_function_that_moves_it_counts_as_an_edit(project):
    found = _source.reachable(project.load().kernel)
    project.write("sd_kernel", "# a new first line\n" + FILES["sd_kernel"])
    with pytest.raises(TilefishBuildError, match="was imported from an older version"):
        _source.check_imported_lines(found.functions)


def test_a_file_that_cannot_be_read_is_a_build_error(project):
    found = _source.reachable(project.load().kernel)
    (project.directory / "sd_kernel.py").unlink()
    with pytest.raises(TilefishBuildError, match="cannot read .*sd_kernel.py"):
        _source.check_imported_lines(found.functions)
