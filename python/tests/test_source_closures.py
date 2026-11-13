"""The source digest covers what a kernel reads from a closure cell and refuses what it cannot stand for.

A kernel made by a factory function (``def make(): S = ...; @triton.jit def k(x): ... S``) reads ``S`` from a closure cell: the
value is in no file, a module namespace does not have it, and the files are the same whatever it is. A digest of files said
the same for ``CLO_S=2.0`` and for ``CLO_S=3.0``, and a build cache keyed by it served the first binary for the second. The digest
now covers the values the kernel's jit functions read that are in no file (their module constants as the process holds them, and
their closure variables) and refuses a value that has no stable ``repr``: ``test_build_cache.py`` runs the same scenario through
``build`` and ``export``/``replay``.

No Triton and no GPU: ``fake_jit.FakeJit`` has what the digest reads of a ``triton.JITFunction``.
"""

import importlib
import linecache
import sys

import pytest

from tilefish import TilefishBuildError, _source

MODULES = ("cl_kernel", "cl_cfg", "cl_helper")

FILES = {
    "cl_kernel": (
        "import os\n"
        "\n"
        "from fake_jit import jit\n"
        "\n"
        "S = 100.0  # a module constant of the same name: the cell is what the kernel reads\n"
        "\n"
        "\n"
        "def make():\n"
        '    S = float(os.environ["CL_S"])\n'
        "\n"
        "    @jit\n"
        "    def k(x):\n"
        "        return x * S\n"
        "\n"
        "    return k\n"
        "\n"
        "\n"
        "k = make()\n"
    ),
    "cl_cfg": "SCALE = 2.0\n",
    "cl_helper": "from fake_jit import jit\n\n\n@jit\ndef double(x):\n    return x * 2\n\n\n@jit\ndef triple(x):\n    return x * 3\n\n\nclass Holder:\n    pass\n",
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
        """Import the project from its files as they are now (the way a new process would), and return the kernel module."""
        for name in MODULES:
            sys.modules.pop(name, None)
        importlib.invalidate_caches()
        linecache.clearcache()
        return importlib.import_module("cl_kernel")


@pytest.fixture
def project(tmp_path, monkeypatch):
    directory = tmp_path / "project"
    directory.mkdir()
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(_source, "triton_version", lambda: "3.5.1")
    monkeypatch.setenv("CL_S", "2.0")
    yield Project(directory)
    for name in MODULES:
        sys.modules.pop(name, None)


def digest(project):
    return _source.source_digest(project.load().k)


# -- the value of a closure variable is part of the digest ------------------------------------------------------------------------


def test_the_value_of_a_closure_variable_is_part_of_the_digest(project, monkeypatch):
    two = digest(project)
    files_of_two = dict(_source.reachable(project.load().k).files)
    monkeypatch.setenv("CL_S", "3.0")
    three = digest(project)
    assert three != two and len(three) == len(two) == 64  # the same files: only the value in the cell differs
    assert dict(_source.reachable(project.load().k).files) == files_of_two
    monkeypatch.setenv("CL_S", "2.0")
    assert digest(project) == two  # and the same value is the same digest, in a new import


def test_a_closure_variable_is_listed_by_the_function_it_belongs_to_and_is_not_the_module_constant_of_its_name(project):
    found = _source.reachable(project.load().k)
    values = _source.values_read(found)
    assert list(values.values()) == ["2.0"]  # S of the cell, not the 100.0 that the module has under the same name
    (key,) = values
    assert key.startswith("cl_kernel:make.<locals>.k@") and key.endswith("/S")
    assert "cl_kernel.S" not in values
    # the module constant is not read by the kernel, so it is not part of what the digest is of: changing it is an edit of the file
    # (it is in the file's text), and assigning it at run time is not seen


def test_the_closure_variables_of_every_kind_of_scalar_are_written_as_their_repr(project, monkeypatch):
    project.write(
        "cl_kernel",
        "from fake_jit import jit\n\n\ndef make(*cells):\n    a, b, c, d, e, f = cells\n\n    @jit\n    def k(x):\n        return (a, b, c, d, e, f, x)\n\n    return k\n\n\n"
        "k = make(1, 1.0, True, 'one', None, (1, 'x'))\n",
    )
    values = _source.values_read(_source.reachable(project.load().k))
    assert sorted(values.values()) == sorted(["1", "1.0", "True", "'one'", "None", "(1, 'x')"])  # 1, 1.0 and True are three values
    assert _source.values_read(_source.reachable(project.load().k), strict=True) == values


def test_a_frozenset_and_a_tuple_are_written_the_same_in_every_process():
    assert _source._canonical(frozenset({"b", "a", "c"})) == "frozenset({'a', 'b', 'c'})"
    assert _source._canonical(frozenset({"c", "b", "a"})) == _source._canonical(frozenset({"a", "b", "c"}))
    assert _source._canonical((1,)) == "(1,)" and _source._canonical(()) == "()" and _source._canonical((1, frozenset({2, 1}))) == "(1, frozenset({1, 2}))"
    assert _source._canonical(float("nan")) == "nan" and _source._canonical(-0.0) == "-0.0"


def test_a_cell_that_has_no_value_yet_is_a_value_too(project):
    project.write(
        "cl_kernel",
        "import os\n\nfrom fake_jit import jit\n\n\ndef make():\n    @jit\n    def k(x):\n        return x * later\n\n"
        '    if os.environ.get("CL_SET"):\n        later = 2.0\n    return k\n\n\nk = make()\n',
    )
    (value,) = _source.values_read(_source.reachable(project.load().k)).values()
    assert value == "<empty cell>"
    assert _source.source_digest(project.load().k) is not None


def test_a_kernel_that_calls_itself_through_its_cell_is_walked_once(project):
    project.write(
        "cl_kernel",
        "from fake_jit import jit\n\n\ndef make():\n    @jit\n    def k(x):\n        return k(x)\n\n    return k\n\n\nk = make()\n",
    )
    found = _source.reachable(project.load().k)
    assert [f.__name__ for f in found.functions] == ["k"]
    assert list(_source.values_read(found).values()) == [f"jit:{_source._function_id(found.functions[0])}"]
    assert _source.source_digest(project.load().k) is not None


# -- what a closure variable holds that is code: the files it comes from are reached, and which one it is is part of the digest ------


def helper_kernel(helper):
    """A kernel module that imports its helper inside the factory: nothing but the cell holds it."""
    return (
        "from fake_jit import jit\n\n\ndef make():\n"
        f"    from cl_helper import {helper} as helper\n\n"
        "    @jit\n    def k(x):\n        return helper(x)\n\n    return k\n\n\nk = make()\n"
    )


def test_a_function_a_kernel_holds_in_a_cell_is_reached(project):
    project.write("cl_kernel", helper_kernel("double"))
    found = _source.reachable(project.load().k)
    assert set(found.files) == {"cl_kernel", "cl_helper", "fake_jit"}  # cl_helper is in no namespace the walk starts from: only the cell has it
    assert sorted(f.__name__ for f in found.functions) == ["double", "k", "triple"]
    with_double = _source.source_digest(project.load().k)
    project.edit("cl_helper", "return x * 2", "return x * 20")
    assert _source.source_digest(project.load().k) != with_double  # an edit of the file of what the cell holds


def test_which_function_is_in_the_cell_is_a_difference_even_when_the_files_are_the_same(project, monkeypatch):
    project.write(
        "cl_kernel",
        "import os\n\nfrom cl_helper import double, triple\nfrom fake_jit import jit\n\n\n"
        "def make(helper):\n    @jit\n    def k(x):\n        return helper(x)\n\n    return k\n\n\n"
        'k = make(double if os.environ["CL_S"] == "2.0" else triple)\n',
    )
    monkeypatch.setenv("CL_S", "2.0")
    with_double = digest(project)
    monkeypatch.setenv("CL_S", "3.0")
    assert digest(project) != with_double  # not one byte of any file differs
    monkeypatch.setenv("CL_S", "2.0")
    assert digest(project) == with_double


def test_a_module_in_a_cell_is_reached_and_what_the_kernel_reads_from_it_is_a_value(project):
    project.write(
        "cl_kernel",
        "from fake_jit import jit\n\n\ndef make():\n    import cl_cfg\n\n    @jit\n    def k(x):\n        return x * cl_cfg.SCALE\n\n    return k\n\n\nk = make()\n",
    )
    found = _source.reachable(project.load().k)
    assert "cl_cfg" in found.files
    values = _source.values_read(found)
    assert values["cl_cfg.SCALE"] == "2.0" and any(v == "module:cl_cfg" for v in values.values())
    before = _source.source_digest(project.load().k)
    project.edit("cl_cfg", "2.0", "4.0")
    assert _source.source_digest(project.load().k) != before


def test_a_constant_a_module_in_a_cell_had_when_it_was_imported_must_still_be_what_it_is(project):
    project.write(
        "cl_kernel",
        "from fake_jit import jit\n\n\ndef make():\n    import cl_cfg\n\n    @jit\n    def k(x):\n        return x * cl_cfg.SCALE\n\n    return k\n\n\nk = make()\n",
    )
    kernel = project.load().k
    assert _source.source_digest(kernel) is not None
    sys.modules["cl_cfg"].SCALE = 9.0  # assigned after it was imported: the binary would be compiled from 9.0
    with pytest.raises(_source.StaleSourceError, match=r"'cl_cfg.SCALE'.*was 2.0, it is 9.0"):
        _source.source_digest(kernel)


# -- what no digest can say: refused, with what to do instead -------------------------------------------------------------------


@pytest.mark.parametrize(
    "cell, kind",
    [
        ("[1, 2]", "list"),
        ("{'a': 1}", "dict"),
        ("cl_helper.Holder()", "Holder"),  # an instance: its repr is its address
        ("object()", "object"),
        ("(1, [2])", "tuple"),  # a tuple that holds a list is not a tuple of scalars
    ],
)
def test_a_closure_variable_that_has_no_stable_repr_is_refused(project, cell, kind):
    project.write(
        "cl_kernel",
        f"import cl_helper\nfrom fake_jit import jit\n\n\ndef make():\n    table = {cell}\n\n    @jit\n    def k(x):\n        return x * table\n\n    return k\n\n\nk = make()\n",
    )
    kernel = project.load().k
    with pytest.raises(_source.UnstableValueError) as e:
        _source.source_digest(kernel)
    message = str(e.value)
    assert "the closure variable 'table' of the Triton function 'k'" in message and kind in message
    assert "make it a module-level constant" in message and "const parameter" in message
    assert isinstance(e.value, TilefishBuildError)
    # the cache key, which has no digest to refuse with, leaves it out instead of making one up
    assert _source.values_read(_source.reachable(kernel)) == {}


def test_a_triton_object_whose_repr_is_its_address_is_refused_not_given_a_made_up_name(project):
    project.write(
        "cl_kernel",
        "import types\n\nfrom fake_jit import jit\n\n\nOpaque = type('Opaque', (), {'__module__': 'triton.language.core'})\n\n\n"
        "def make():\n    dtype = Opaque()\n\n    @jit\n    def k(x):\n        return x * dtype\n\n    return k\n\n\nk = make()\n",
    )
    kernel = project.load().k
    with pytest.raises(_source.UnstableValueError, match="closure variable 'dtype'"):
        _source.source_digest(kernel)
    assert list(_source.values_read(_source.reachable(kernel)).values()) == ["Opaque"]  # a cache key falls back to the type's name
    # one whose repr says what it is (``constexpr[2.0]``) is a value like a scalar
    project.write(
        "cl_kernel",
        "from fake_jit import jit\n\n\nclass Constexpr:\n    __module__ = 'triton.language.core'\n\n    def __repr__(self):\n        return 'constexpr[2.0]'\n\n\n"
        "def make():\n    c = Constexpr()\n\n    @jit\n    def k(x):\n        return x * c\n\n    return k\n\n\nk = make()\n",
    )
    kernel = project.load().k
    assert list(_source.values_read(_source.reachable(kernel), strict=True).values()) == ["constexpr[2.0]"]
    assert _source.source_digest(kernel) is not None


def test_a_module_attribute_a_kernel_reads_that_has_no_stable_repr_is_refused_and_one_it_does_not_read_is_not(project):
    project.write("cl_cfg", "SCALE = 2.0\nTABLE = [1, 2]\n")
    project.write(
        "cl_kernel",
        "import cl_cfg\nfrom fake_jit import jit\n\n\n@jit\ndef k(x):\n    return x * cl_cfg.SCALE\n",
    )
    assert _source.source_digest(project.load().k) is not None  # TABLE is in the module, and nothing reads it
    project.write(
        "cl_kernel",
        "import cl_cfg\nfrom fake_jit import jit\n\n\n@jit\ndef k(x):\n    return x * cl_cfg.TABLE\n",
    )
    with pytest.raises(_source.UnstableValueError, match=r"the module attribute 'cl_cfg.TABLE', which the Triton function 'k' reads, holds a list"):
        _source.source_digest(project.load().k)


def test_functions_classes_and_modules_that_a_kernel_reads_are_code_not_values(project):
    project.write(
        "cl_kernel",
        "import cl_helper\nimport math\nfrom cl_helper import Holder, double\nfrom fake_jit import jit\n\n\n@jit\ndef k(x):\n    return double(x) + cl_helper.triple(x) + math.pi + Holder\n",
    )
    kernel = project.load().k
    assert _source.values_read(_source.reachable(kernel), strict=True) == {}  # all of them are in files
    assert _source.source_digest(kernel) is not None
