"""Gluon kernels as externs, on a real GPU (``@tilefish.extern(gluon=...)``, the kernels of ``gluon_kernels.py``).

A Gluon kernel pins its thread mapping: which thread holds which element, so the order of its reductions. Through tilefish it is built by Gluon's
frontend (``GluonASTSource``), identified by the TTGIR that frontend builds (``source_digest_kind`` ``"tilefish.gluon-ir.v1"``), and runs in ``jit``,
``replay`` and in a plan exactly like a ``triton.jit`` kernel: ``jit`` and ``replay`` give the same bytes for every bucket, with the views of one
buffer read through one ``i32`` row stride and offsets in ``i64``; the contraction it is declared with is in its binary; a layout written for four
warps runs the same per-row order in a kernel of eight; a layout compiled for a number of warps it is not written for is refused. And a Gluon kernel
that was edited (its layout, or a constant of its body) is refused by a ``replay`` in a new process, with a pin and without one
(``gluon_child.py`` is that process). ``rust_gate.py`` holds the Rust host to the same bytes.
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
pytest.importorskip("triton.experimental.gluon")
pytest.importorskip("cuda.bindings.driver")

import gluon_child  # noqa: E402
import gluon_kernels as gk  # noqa: E402
from gpu_util import differing, randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishBuildError, TilefishDeclarationError, TilefishPreconditionError, _ir, runtime  # noqa: E402
from tilefish.aot import compile_specs  # noqa: E402

F32 = torch.float32
CAP = torch.cuda.get_device_capability()
ARCH = f"sm_{CAP[0] * 10 + CAP[1]}"
TESTS = pathlib.Path(__file__).resolve().parent
SRC = pathlib.Path(tilefish.__file__).resolve().parent.parent
OPS = (gk.row_dot, gk.row_dot_fma, gk.row_sum4, gk.row_sum8)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("gluon")
    manifest = runtime.export(list(OPS), directory)
    return directory, manifest


def run_both(op, exported, *args):
    directory, manifest = exported
    jit = tilefish.run(op, *args)
    replay = tilefish.run(op, *args, mode="replay", manifest=directory, pin=manifest.digest)
    assert same_bytes(jit, replay), f"{op.name}: jit and replay differ"
    return replay


def halves(rows, c, seed):
    """``a`` and ``b``: the two halves of one ``[rows, 2, c]`` tensor, views with the row stride ``2 * c`` (offsets 0 and ``c``)."""
    pair = randn((rows, 2, c), F32, seed=seed)
    return pair[:, 0], pair[:, 1]


def close(got, a, b):
    want = gk.row_dot_reference(a, b)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), (got - want).abs().max()


def test_the_manifest_records_the_gluon_ir_of_each_binary(exported):
    _, manifest = exported
    for op in (gk.row_dot, gk.row_dot_fma):
        for spec in compile_specs(op):
            entry = manifest.get(spec.stem, ARCH).entry
            assert entry["kind"] == "triton" and entry["source_digest_kind"] == "tilefish.gluon-ir.v1"
            inputs = spec.ast_source_inputs()
            text = _ir.ttir_text(gk.row_dot_kernel, inputs, arch=CAP[0] * 10 + CAP[1], options=inputs.options)
            assert entry["source_digest"] == _ir.ir_digest(text, "gluon")
            assert '"ttg.num-warps" = 4' in text and "#ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1]" in text
            (s_row,) = [p for p in entry["abi"]["params"] if p["name"] == "s_row"]
            assert (s_row["type"], s_row["ptx_type"]) == ("i32", "u32") and len(s_row["stride_of"]) == 2
            assert entry["options"]["enable_fp_fusion"] is (op is gk.row_dot_fma)
            assert not [p for p in entry["preconditions"] if p["kind"] in ("max_elements", "max_extent")]  # index_bits=64


@pytest.mark.parametrize("c", [128, 256])
@pytest.mark.parametrize("rows", [1, 7, 40, 1000])
def test_jit_and_replay_are_byte_identical_over_two_views_of_one_buffer(exported, rows, c):
    a, b = halves(rows, c, seed=rows + c)
    assert a.stride() == b.stride() == (2 * c, 1) and b.storage_offset() == c
    for op in (gk.row_dot, gk.row_dot_fma):
        close(run_both(op, exported, a, b), a, b)


@pytest.mark.parametrize("c", [128, 256])
def test_two_tensors_of_their_own_with_the_same_stride_are_fine_and_another_stride_is_refused(exported, c):
    a, b = randn((9, c), F32, seed=1), randn((9, c), F32, seed=2)
    close(run_both(gk.row_dot, exported, a, b), a, b)
    directory, manifest = exported
    wide = halves(9, c, seed=3)[0]  # row stride 2c, where b has c
    for mode, extra in (("jit", {}), ("replay", {"manifest": directory, "pin": manifest.digest})):
        with pytest.raises(TilefishPreconditionError, match=rf"argument 'b'.*the stride of axis 0 is {c}, but the signature requires s_row = {2 * c}") as e:
            tilefish.run(gk.row_dot, wide, b, mode=mode, **extra)
        assert e.value.kind == "stride"


def test_the_contraction_it_is_declared_with_is_in_its_binary(exported):
    _, manifest = exported
    for c in (128, 256):
        plain, fused = manifest.get(f"row_dot.Cb{c}", ARCH).entry, manifest.get(f"row_dot_fma.Cb{c}", ARCH).entry
        assert plain["sha256_code"] != fused["sha256_code"]
    a, b = halves(1000, 256, seed=11)
    plain, fused = run_both(gk.row_dot, exported, a, b), run_both(gk.row_dot_fma, exported, a, b)
    assert differing(plain, fused) > 0  # one FMA per element rounds once where a multiply and an add round twice


def test_a_layout_pins_the_order_whatever_the_number_of_warps(exported):
    """``row_sum4`` and ``row_sum8`` give each row to one warp with the same per-thread layout: the same order, the same bits."""
    for rows in (1, 7, 40, 1001):
        x = randn((rows, 256), F32, seed=rows)
        four, eight = run_both(gk.row_sum4, exported, x), run_both(gk.row_sum8, exported, x)
        assert same_bytes(four, eight)
        assert torch.allclose(four, x.double().sum(-1).float(), rtol=1e-5, atol=1e-5)


def test_a_layout_compiled_for_another_number_of_warps_is_refused():
    x = randn((8, 256), F32, seed=0)
    with pytest.raises(TilefishBuildError, match="could not compile 'row_sum_wrong_warps'"):
        tilefish.run(gk.row_sum_wrong_warps, x)


def test_the_frontend_of_another_language_is_refused():
    with pytest.raises(TilefishDeclarationError, match="is a Gluon kernel .*declare it with gluon="):
        tilefish.extern(triton=gk.row_sum_kernel, grid=lambda: (1,))
    with pytest.raises(TilefishDeclarationError, match="gluon= takes a Gluon kernel .*declare it with triton="):
        tilefish.extern(gluon=gk.row_sum_triton, grid=lambda: (1,))
    R = tilefish.Dim("R")
    x = randn((8, 256), F32, seed=0)
    for kind, ref, message in (
        ("gluon", "gluon_kernels:row_sum_triton", "is declared a Gluon kernel .* but it is a triton.jit function"),
        ("triton", "gluon_kernels:row_sum_kernel", r"is a Gluon kernel \(@gluon.jit\), but it is declared a triton.jit kernel: declare it with @tilefish.extern\(gluon="),
    ):

        @tilefish.extern(**{kind: ref}, num_warps=4, grid=lambda R: (tilefish.cdiv(R, 4),), index_bits=64, name=f"wrong_{kind}")
        def wrong(
            x: tilefish.Strided[tilefish.Tensor[R, 256, tilefish.f32]],
            y: tilefish.Out[tilefish.Tensor[R, tilefish.f32]],
            s_row: tilefish.StrideOf("x", 0, dtype=tilefish.i32),
            n_rows: R,
            C: tilefish.const = 256,
            WARPS: tilefish.const = 4,
        ): ...

        with pytest.raises(TilefishBuildError, match=message):
            tilefish.run(wrong, x)


@pytest.fixture(scope="module")
def exported_plan(tmp_path_factory):
    directory = tmp_path_factory.mktemp("gluon-plan")
    return directory, runtime.export(gk.halves_dot, directory)


@pytest.mark.parametrize("rb", [8, 24])
def test_a_plan_runs_a_gluon_step_over_two_views_of_an_intermediate(exported_plan, rb):
    directory, plan = exported_plan
    x = randn((rb, 256), F32, seed=rb)
    replay = tilefish.run(gk.halves_dot, x, mode="replay", plan=directory, pin=plan.digest)
    jit = tilefish.run(gk.halves_dot, x)
    pair = tilefish.run(gk.doubled, x).view(rb, 2, 128)  # the per-op eager composition
    eager = tilefish.run(gk.row_dot, pair[:, 0], pair[:, 1])
    assert same_bytes(replay, jit) and same_bytes(replay, eager)
    reference = tilefish.run(gk.halves_dot, x, mode="reference")
    assert torch.allclose(replay, reference, rtol=1e-5, atol=1e-5)
    captured = tilefish.capture(gk.halves_dot, x, mode="replay", plan=directory, pin=plan.digest)
    captured.replay()
    torch.cuda.synchronize()
    assert same_bytes(captured.outputs, replay)
    (step,) = [s for s in plan.steps(Rb=rb) if s["entry"] == "row_dot.Cb128"]
    assert [(a["view"]["strides"], a["view"]["offset"]) for a in step["args"][:2]] == [([256, 1], 0), ([256, 1], 128)]


# -- an edited Gluon kernel, in new processes ----------------------------------------------------------------------------------------------


def child(directory, role, export_dir, pin=""):
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join([str(SRC), str(TESTS), env.get("PYTHONPATH", "")]))
    r = subprocess.run(
        [sys.executable, str(TESTS / "gluon_child.py"), str(directory), role, str(export_dir), pin],
        env=env,
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=900,
    )
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert r.returncode == 0 and lines, f"{role} failed:\n{r.stderr[-3000:]}"
    return json.loads(lines[-1][len("RESULT ") :])


def test_an_edited_gluon_kernel_is_refused_by_a_replay_in_a_new_process_pinned_or_not(tmp_path):
    """Process A exports the kernel; the file is edited (its layout: four columns per thread become two; or a constant of its body); process B
    imports the edited file and replays A's manifest with A's pin and with ``trust_directory=True``: both are refused as stale. With the file as it
    was, process C replays it, pinned and not, and gets A's bytes."""
    project, export_dir = tmp_path / "project", tmp_path / "export"
    gluon_child.write(project)
    a = child(project, "export", export_dir)
    entry = json.loads((export_dir / "manifest.json").read_text())["kernels"][0]
    assert entry["source_digest_kind"] == "tilefish.gluon-ir.v1" and entry["source_digest"] == a["digest"]
    for edit in (dict(per_thread=2), dict(scale="2.0")):
        gluon_child.write(project, **edit)
        b = child(project, "replay", export_dir, pin=a["pin"])
        assert b["digest"] != a["digest"]
        for how in ("pinned", "trusted"):
            refused = b[how].get("refused") or ""
            assert "built from Gluon IR with digest" in refused and "stale" in refused, (edit, how, b[how])
    gluon_child.write(project)
    c = child(project, "replay", export_dir, pin=a["pin"])
    assert c["digest"] == a["digest"]
    assert c["pinned"] == c["trusted"] == {"replay": a["jit"]}
