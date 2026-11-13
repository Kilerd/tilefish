"""An op with an implementation per architecture, on a real GPU (``@tilefish.extern(by_arch=...)``, the kernels of ``per_arch_kernels.py``).

``row_sum`` declares two tilings of one row reduction, one for sm_103 and one for sm_120 (and as the fallback). On the device, ``export`` builds the
implementation of the device's architecture and records it in the manifest entry; ``jit`` and ``replay`` run it and give the same bytes, which are
those of that tiling declared alone (and, for most rows, not those of the other tiling); a plan runs it too. A changed implementation for this
architecture is stale: its options (in process) and its kernel (``per_arch_child.py``, in new processes, pinned or not); a change of the other
architecture's kernel leaves this export valid. ``rust_gate.py`` holds the Rust host to the same bytes.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import per_arch_child  # noqa: E402
import per_arch_kernels as pk  # noqa: E402
from gpu_util import differing, randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Out, Tensor, TilefishManifestError, TilefishRuntimeError, const, f32, runtime  # noqa: E402
from tilefish.aot import compile_specs, declaration_digest  # noqa: E402

F32 = torch.float32
CAP = torch.cuda.get_device_capability()
CC = CAP[0] * 10 + CAP[1]
ARCH = f"sm_{CC}"
HERE = pk.row_sum.for_arch(CC)  # the implementation this device runs
OTHER = pk.whole_only if HERE.implementation == "sm_103" else pk.tiled_only  # the other tiling, declared alone
SAME = pk.tiled_only if HERE.implementation == "sm_103" else pk.whole_only  # this tiling, declared alone
TESTS = pathlib.Path(__file__).resolve().parent
SRC = pathlib.Path(tilefish.__file__).resolve().parent.parent
ROWS = (1, 2, 7, 40, 1000)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("per_arch")
    return directory, runtime.export([pk.row_sum, pk.tiled_only, pk.whole_only], directory)


def test_the_export_builds_the_implementation_of_the_devices_architecture(exported):
    _, manifest = exported
    entry = manifest.get("row_sum", ARCH).entry
    (spec,) = compile_specs(pk.row_sum, arch=CC)
    assert entry["implementation"] == HERE.implementation == spec.implementation
    assert entry["declaration_digest"] == declaration_digest(HERE) == spec.declaration_digest
    assert (entry["num_warps"], entry["grid"]) == (spec.num_warps, spec.grid)
    if CC == 103:
        assert (entry["implementation"], entry["num_warps"], entry["grid"]) == ("sm_103", 4, "(cdiv(R, 2),)")
        assert [(c["name"], c["value"]) for c in entry["constexprs"]] == [("ROWS", 2), ("C", 1024), ("BLOCK", 128)]
    # the machine code is this tiling's (declared alone: the same kernel, constants and warps) and not the other's
    assert entry["sha256_code"] == manifest.get(SAME.name, ARCH).sha256_code != manifest.get(OTHER.name, ARCH).sha256_code


@pytest.mark.parametrize("rows", ROWS)
def test_jit_and_replay_run_that_implementation_and_give_its_bytes(exported, rows):
    directory, manifest = exported
    x = randn((rows, 1024), F32, seed=rows)
    jit = tilefish.run(pk.row_sum, x)
    replay = tilefish.run(pk.row_sum, x, mode="replay", manifest=directory, pin=manifest.digest)
    assert same_bytes(jit, replay)
    assert same_bytes(replay, tilefish.run(SAME, x, mode="replay", manifest=directory, pin=manifest.digest))
    assert torch.allclose(replay, x.double().sum(-1).float(), rtol=1e-5, atol=1e-3)
    assert same_bytes(tilefish.run(pk.row_sum, x, mode="reference"), pk.row_sum_reference(x))  # one reference, whatever the architecture


def test_the_two_tilings_are_other_bits(exported, capsys):
    """Why the choice matters: the other tiling adds the same 1024 products in another order."""
    directory, manifest = exported
    x = randn((1000, 1024), F32, seed=5)
    here = tilefish.run(pk.row_sum, x, mode="replay", manifest=directory, pin=manifest.digest)
    other = tilefish.run(OTHER, x, mode="replay", manifest=directory, pin=manifest.digest)
    n = differing(here, other)
    with capsys.disabled():
        print(f"\n{ARCH}: row_sum runs the implementation {HERE.implementation!r}; {n} of 1000 rows differ from the other tiling's bits")
    assert n > 0


def test_a_plan_runs_the_implementation_of_its_architecture(tmp_path):
    p = runtime.export(pk.scaled_row_sum, tmp_path / "plan")
    assert p.arch == ARCH and [st["entry"] for st in p.steps(Rb=8)] == ["doubled", "row_sum"]
    entry = tilefish.Manifest.load(tmp_path / "plan").get("row_sum", ARCH).entry
    assert entry["implementation"] == HERE.implementation
    for rb in (8, 24):
        x = randn((rb, 1024), F32, seed=rb)
        jit = tilefish.run(pk.scaled_row_sum, x)
        replay = tilefish.run(pk.scaled_row_sum, x, mode="replay", plan=tmp_path / "plan", pin=p.digest)
        assert same_bytes(jit, replay) and same_bytes(replay, tilefish.run(SAME, x * 2.0))


def test_an_architecture_without_an_implementation_is_refused(tmp_path):
    R = Dim("R")
    other = "sm_90" if CC != 90 else "sm_80"

    @tilefish.extern(by_arch={other: tilefish.impl(triton=pk.row_sum_whole, num_warps=8)}, grid=lambda R: (R,))
    def elsewhere(x: Tensor[R, 1024, f32], y: Out[Tensor[R, f32]], n_rows: R, ROWS: const = 1, C: const = 1024, BLOCK: const = 1024): ...

    x = randn((4, 1024), F32, seed=0)
    with pytest.raises(TilefishRuntimeError, match=f"no implementation for {ARCH}"):
        tilefish.run(elsewhere, x)
    with pytest.raises(TilefishRuntimeError, match=f"no implementation for {ARCH}"):
        runtime.export(elsewhere, tmp_path / "none")
    assert not (tmp_path / "none" / "manifest.json").exists()


def test_a_changed_implementation_of_this_architecture_is_refused(exported):
    """In process: the declaration changes what this architecture runs (its options; or the map sends it to the fallback). Replay refuses, pinned."""
    directory, manifest = exported
    x = randn((7, 1024), F32, seed=1)
    R = Dim("R")
    here = HERE.implementation
    for changed_impl, fallback in (
        (tilefish.impl(triton=pk.row_sum_tiled if here == "sm_103" else pk.row_sum_whole, num_warps=2, consts=dict(pk.TILED.consts if here == "sm_103" else {})), None),
        (None, tilefish.impl(triton=pk.row_sum_tiled if here == "sm_103" else pk.row_sum_whole, num_warps=4)),
    ):
        by_arch = {"sm_103": pk.TILED, "sm_120": pk.WHOLE}
        if changed_impl is None:
            by_arch.pop(here)
        else:
            by_arch[here] = changed_impl

        @tilefish.extern(by_arch=by_arch, fallback=fallback, grid=lambda R, ROWS: (tilefish.cdiv(R, ROWS),), name="row_sum")
        def row_sum(x: Tensor[R, 1024, f32], y: Out[Tensor[R, f32]], n_rows: R, ROWS: const = 1, C: const = 1024, BLOCK: const = 1024): ...

        with pytest.raises(TilefishManifestError, match="stale|first difference at"):
            tilefish.run(row_sum, x, mode="replay", manifest=directory, pin=manifest.digest)


# -- an edited kernel, in new processes -------------------------------------------------------------------------------------------------------


def child(directory, role, export_dir, pin=""):
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join([str(SRC), str(TESTS), env.get("PYTHONPATH", "")]))
    r = subprocess.run(
        [sys.executable, str(TESTS / "per_arch_child.py"), str(directory), role, str(export_dir), pin],
        env=env,
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=900,
    )
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert r.returncode == 0 and lines, f"{role} failed:\n{r.stderr[-3000:]}"
    return json.loads(lines[-1][len("RESULT ") :])


def test_an_edited_kernel_of_this_architecture_is_stale_and_one_of_another_is_not(tmp_path, capsys):
    """Process A exports the op on this device. The kernel of this architecture's implementation is edited (a constant of its body): process B
    replays A's manifest with A's pin and with ``trust_directory=True`` and both are refused as stale. The kernel of the other architecture's
    implementation edited instead: process C replays A's manifest, pinned and not, and gets A's bytes."""
    project, export_dir = tmp_path / "project", tmp_path / "export"
    per_arch_child.write(project)
    a = child(project, "export", export_dir)
    entry = json.loads((export_dir / "manifest.json").read_text())["kernels"][0]
    assert a["implementation"] == entry["implementation"] == HERE.implementation and entry["source_digest"] == a["digest"]
    mine, theirs = ("tiled", "whole") if HERE.implementation == "sm_103" else ("whole", "tiled")
    per_arch_child.write(project, **{mine: "2.0"})
    b = child(project, "replay", export_dir, pin=a["pin"])
    assert b["digest"] != a["digest"]
    for how in ("pinned", "trusted"):
        refused = b[how].get("refused") or ""
        assert "built from Triton IR with digest" in refused and "stale" in refused, (how, b[how])
    per_arch_child.write(project, **{theirs: "2.0"})
    c = child(project, "replay", export_dir, pin=a["pin"])
    assert c["digest"] == a["digest"]
    assert c["pinned"] == c["trusted"] == {"replay": a["jit"]}
    with capsys.disabled():
        print(f"\n{ARCH}: export pin {a['pin'][:16]}..., {HERE.implementation} digest {a['digest'][:16]}...; its kernel edited: digest "
              f"{b['digest'][:16]}..., refused pinned and trusted; the other kernel edited: replay {c['pinned']['replay'][:16]}... == jit {a['jit'][:16]}...")
