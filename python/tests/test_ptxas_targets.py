"""The helper of the GPU tests that build for architectures other than the device's (``conftest.ptxas_problem``), with no GPU and no Triton.

Triton 3.5.1 bundles the ptxas of CUDA 12.8, which cannot target sm_103a. The helper reads which ``--gpu-name`` values the ptxas Triton compiles
with takes (from ``ptxas --help``), maps a compute capability to the target Triton hands ptxas, and gives the reason a test skips: the ptxas, its
version, and ``TRITON_PTXAS_PATH``. Here on the text the two ptxas releases print, and on a stand-in ptxas.
"""

import os
import stat
import sys

import pytest

import conftest
from conftest import ptxas_gpu_names, ptxas_problem, ptxas_release, require_ptxas_for, triton_gpu_name

# what `ptxas --help` prints around `--gpu-name`, for CUDA 12.8 (the ptxas bundled with Triton 3.5.1) and CUDA 12.9
HELP_12_8 = """\
--generate-line-info                                (-lineinfo)
        Generate line-number information for device code.

--gpu-name <gpu name>                               (-arch)
        Specify name of NVIDIA GPU to generate code for. This option also takes virtual
        compute architectures, in which case code generation is suppressed. This
        can be used for parsing only.
        Allowed values for this option:  'compute_100','compute_100a','compute_101',
        'compute_101a','compute_120','compute_120a','compute_50','compute_52','compute_53',
        'compute_60','compute_61','compute_62','compute_70','compute_72','compute_75',
        'compute_80','compute_86','compute_87','compute_89','compute_90','compute_90a',
        'lto_100','lto_100a','lto_101','lto_101a','lto_120','lto_120a','lto_50',
        'lto_52','lto_53','lto_60','lto_61','lto_62','lto_70','lto_72','lto_75',
        'lto_80','lto_86','lto_87','lto_89','lto_90','lto_90a','sm_100','sm_100a',
        'sm_101','sm_101a','sm_120','sm_120a','sm_50','sm_52','sm_53','sm_60','sm_61',
        'sm_62','sm_70','sm_72','sm_75','sm_80','sm_86','sm_87','sm_89','sm_90',
        'sm_90a'.
        Default value:  'sm_52'.

--help                                              (-h)
        Print this help information on this tool.

--machine <bits>                                    (-m)
        Specify 'address size' in bits.
        Allowed values for this option:  64.
        Default value:  64.
"""

HELP_12_9 = """\
--gpu-name <gpu name>                               (-arch)
        Specify name of NVIDIA GPU to generate code for. This option also takes virtual
        compute architectures, in which case code generation is suppressed. This
        can be used for parsing only.
        Allowed values for this option:  'compute_100','compute_100a','compute_100f',
        'compute_101','compute_101a','compute_101f','compute_103','compute_103a',
        'compute_103f','compute_120','compute_120a','compute_120f','compute_121',
        'compute_121a','compute_121f','compute_50','compute_52','compute_53','compute_60',
        'compute_61','compute_62','compute_70','compute_72','compute_75','compute_80',
        'compute_86','compute_87','compute_89','compute_90','compute_90a','lto_100',
        'lto_100a','lto_100f','lto_101','lto_101a','lto_101f','lto_103','lto_103a',
        'lto_103f','lto_120','lto_120a','lto_120f','lto_121','lto_121a','lto_121f',
        'lto_50','lto_52','lto_53','lto_60','lto_61','lto_62','lto_70','lto_72',
        'lto_75','lto_80','lto_86','lto_87','lto_89','lto_90','lto_90a','sm_100',
        'sm_100a','sm_100f','sm_101','sm_101a','sm_101f','sm_103','sm_103a','sm_103f',
        'sm_120','sm_120a','sm_120f','sm_121','sm_121a','sm_121f','sm_50','sm_52',
        'sm_53','sm_60','sm_61','sm_62','sm_70','sm_72','sm_75','sm_80','sm_86',
        'sm_87','sm_89','sm_90','sm_90a'.
        Default value:  'sm_52'.

--help                                              (-h)
        Print this help information on this tool.
"""

VERSION_12_8 = """\
ptxas: NVIDIA (R) Ptx optimizing assembler
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:21:21_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
"""


def test_the_targets_of_a_ptxas_are_read_from_its_help():
    old, new = ptxas_gpu_names(HELP_12_8), ptxas_gpu_names(HELP_12_9)
    assert len(old) == 3 * 21 and len(new) == 3 * 30  # compute_, lto_ and sm_ of each architecture
    assert {"sm_80", "sm_90a", "sm_100a", "sm_120a", "compute_120a", "lto_90a"} <= old
    assert "sm_103a" not in old and "sm_103" not in old and "sm_121a" not in old
    assert {"sm_103", "sm_103a", "sm_103f", "sm_121a"} <= new and old < new
    # the values of another option (`--machine`: 64) are not targets, and a text without the list has none
    assert "64" not in old
    assert ptxas_gpu_names("") == frozenset() and ptxas_gpu_names("--gpu-name <gpu name>\n  Specify the GPU.\n") == frozenset()


def test_the_target_triton_builds_for_is_architecture_specific_from_sm_90_on():
    assert [triton_gpu_name(c) for c in (75, 80, 86, 89, 90, 100, 103, 120, 121)] == [
        "sm_75",
        "sm_80",
        "sm_86",
        "sm_89",
        "sm_90a",
        "sm_100a",
        "sm_103a",
        "sm_120a",
        "sm_121a",
    ]


def test_the_version_of_a_ptxas_is_read_from_its_version_line():
    assert ptxas_release(VERSION_12_8) == "12.8.93"
    assert ptxas_release("ptxas 99\n") == "ptxas 99" and ptxas_release("") == "of unknown version"


@pytest.fixture
def fake_ptxas(tmp_path, monkeypatch):
    """A stand-in ptxas that prints ``help`` for ``--help`` and the 12.8 version for ``--version``, as the one Triton compiles with."""

    def make(help_text):
        directory = tmp_path / f"ptxas{len(list(tmp_path.iterdir()))}"
        directory.mkdir()
        (directory / "help.txt").write_text(help_text)
        (directory / "version.txt").write_text(VERSION_12_8)
        script = directory / "ptxas"
        script.write_text(
            f"#!{sys.executable}\nimport sys, pathlib\n"
            f"d = pathlib.Path({str(directory)!r})\n"
            "sys.stdout.write((d / ('version.txt' if sys.argv[1:] == ['--version'] else 'help.txt')).read_text())\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setattr(conftest, "triton_ptxas", lambda: str(script))
        return str(script)

    return make


@pytest.mark.skipif(os.name == "nt", reason="the stand-in ptxas is a script with a shebang")
def test_a_test_that_needs_a_target_the_ptxas_does_not_have_skips_with_what_to_do(fake_ptxas):
    path = fake_ptxas(HELP_12_8)
    assert ptxas_problem(80, "sm_90", 100, 120) is None
    assert ptxas_problem(103) == (
        f"the ptxas Triton compiles with ({path}, version 12.8.93) cannot target sm_103a: set TRITON_PTXAS_PATH to a ptxas of CUDA 12.9 or newer"
    )
    assert "cannot target sm_103a, sm_121a:" in ptxas_problem("sm_80", "sm_103", 120, 121, 103)  # each once
    with pytest.raises(pytest.skip.Exception, match=r"cannot target sm_103a: set TRITON_PTXAS_PATH"):
        require_ptxas_for(100, 103)
    require_ptxas_for("sm_100", "sm_120")  # nothing missing: no skip
    fake_ptxas(HELP_12_9)
    assert ptxas_problem(80, 90, 100, 103, 120, 121) is None


@pytest.mark.skipif(os.name == "nt", reason="the stand-in ptxas is a script with a shebang")
def test_a_ptxas_whose_help_has_no_list_of_targets_does_not_skip(fake_ptxas):
    """When the targets cannot be read, the test runs: the build says what is wrong."""
    fake_ptxas("usage: ptxas [options] file\n")
    assert ptxas_problem(103) is None


def test_a_ptxas_that_cannot_be_run_or_none_at_all_skips(tmp_path, monkeypatch):
    missing = str(tmp_path / "no-ptxas")
    monkeypatch.setattr(conftest, "triton_ptxas", lambda: missing)
    problem = ptxas_problem(103)
    assert problem.startswith(f"the ptxas Triton compiles with ({missing}) cannot be run: ") and problem.endswith("a ptxas of CUDA 12.9 or newer")
    monkeypatch.setattr(conftest, "triton_ptxas", lambda: None)
    assert ptxas_problem(90) == "there is no ptxas to build with: install Triton, or set TRITON_PTXAS_PATH to a ptxas of CUDA 12.9 or newer"
