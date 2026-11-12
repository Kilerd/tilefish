"""What a replay must not do, on a real GPU: run stale code, trust a directory blindly, launch what does not fit.

* A kernel redeclared with another body but the same name and signature does not replay the old manifest's binary:
  a generated kernel carries the sha256 of its Triton source in its spec, a hand-written one the digest of its files
  (the module of the kernel and every user module it reaches, plus the Triton version) in its manifest entry, which a
  replay recomputes (or refuses, unless the manifest is pinned). What Triton's ``cache_key`` does not see (a constexpr
  read as a module attribute, a dtype alias, a constant in a helper module two levels away) is covered, and no build
  reads Triton's disk cache.
* ``build`` refuses source that is not the spec's: a stale generated file on ``sys.path``, another function passed as
  ``fn``, a module whose file was edited after the process imported it.
* A manifest can be pinned by its digest; the architecture it runs on is the device's; ``shared`` is checked against
  the cubin and the device before a launch that would fault and poison the context; a launch is checked against the
  ABI and the device's limits; ``export`` merges into an existing directory.
* Mutants of the runtime's checks are killed here (each ``test_..._is_checked`` names the check it trips).
"""

import dataclasses
import hashlib
import importlib
import json
import linecache
import os
import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

from gpu_util import randn, same_bytes  # noqa: E402
from lower_cases import gelu_exact, softmax  # noqa: E402

import tilefish  # noqa: E402
from tilefish import (  # noqa: E402
    Dim,
    Manifest,
    Out,
    Tensor,
    TilefishBuildError,
    TilefishManifestError,
    TilefishPreconditionError,
    TilefishRuntimeError,
    _cuda,
    f32,
    runtime,
)
from tilefish import _source  # noqa: E402
from tilefish._elf import parse_elf  # noqa: E402
from tilefish.aot import build  # noqa: E402
from tilefish.dims import as_dim, cdiv  # noqa: E402
from tilefish.signature import Param, Signature  # noqa: E402

CAP = torch.cuda.get_device_capability()
ARCH = f"sm_{CAP[0] * 10 + CAP[1]}"
BF16 = torch.bfloat16


def arange_rows():
    return torch.arange(128, dtype=torch.float32, device="cuda").reshape(2, 64)


# -- H1: a kernel whose body changed ----------------------------------------------------------------------------------------------------


def scale_kernel(factor):
    """The same kernel (name and signature) with another body."""
    M, D = Dim("M"), Dim("D", 64)

    @tilefish.kernel(name="scale_k")
    def scale_k(x: Tensor[M, D, f32]) -> Tensor[M, D, f32]:
        return x * factor

    return scale_k


def test_a_generated_kernel_with_another_body_does_not_replay_the_old_binary(tmp_path):
    old, new = scale_kernel(2.0), scale_kernel(3.0)
    x = arange_rows()
    directory = runtime.export(old, tmp_path).directory
    assert torch.equal(tilefish.run(old, x, mode="replay", manifest=directory), x * 2)  # the manifest is right for its kernel
    low_old, low_new = old.lower(), new.lower()
    assert low_old.source_sha256 != low_new.source_sha256
    with pytest.raises(TilefishManifestError) as e:  # ... and refused for the redeclared one, instead of returning x * 2
        tilefish.run(new, x, mode="replay", manifest=directory)
    assert low_old.source_sha256 in str(e.value) and low_new.source_sha256 in str(e.value)
    assert "changed since the manifest was written" in str(e.value) and "export it again" in str(e.value)
    assert torch.equal(tilefish.run(new, x, mode="jit"), x * 3)  # jit builds the new source
    assert torch.equal(tilefish.run(new, x, mode="replay", manifest=runtime.export(new, tmp_path / "new").directory), x * 3)


EXT_SOURCE = """
import triton
import triton.language as tl


@triton.jit
def ext_scale_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x * {factor}, mask=mask)
"""

Count = Dim("Count", divisible_by=16)


@tilefish.extern(triton="ext_scale_mod:ext_scale_kernel", num_warps=4, grid=lambda Count: (cdiv(Count, 256),))
def ext_scale(x_ptr: Tensor[Count, f32], y_ptr: Out[Tensor[Count, f32]], n_elements: Count, *, BLOCK: tilefish.const = 256): ...


def write_ext_module(directory, factor):
    (directory / "ext_scale_mod.py").write_text(EXT_SOURCE.format(factor=factor))
    sys.modules.pop("ext_scale_mod", None)
    importlib.invalidate_caches()
    linecache.clearcache()
    return importlib.import_module("ext_scale_mod")


@pytest.fixture
def ext_dir(tmp_path, monkeypatch):
    d = tmp_path / "ext"
    d.mkdir()
    monkeypatch.syspath_prepend(str(d))
    yield d
    sys.modules.pop("ext_scale_mod", None)


def test_a_hand_written_kernel_with_another_body_does_not_replay_the_old_binary(ext_dir, tmp_path):
    x = randn((64,), torch.float32, 0)
    write_ext_module(ext_dir, "2.0")
    manifest = runtime.export(ext_scale, tmp_path / "m")
    entry = json.loads((manifest.directory / "manifest.json").read_text())["kernels"][0]
    assert entry["source_digest_kind"] == _source.SOURCE_DIGEST_KIND and len(entry["source_digest"]) == 64
    assert "source_sha256" not in entry  # an extern's declaration contains no source
    assert torch.equal(tilefish.run(ext_scale, x, mode="replay", manifest=manifest.directory), x * 2)
    assert torch.equal(tilefish.run(ext_scale, x, mode="jit"), x * 2)

    # the same file again: the same key, still fine (a reload is not a change)
    write_ext_module(ext_dir, "2.0")
    assert torch.equal(tilefish.run(ext_scale, x, mode="replay", manifest=manifest.directory), x * 2)

    # the body changes, the name and the signature do not
    write_ext_module(ext_dir, "3.0")
    with pytest.raises(TilefishManifestError) as e:
        tilefish.run(ext_scale, x, mode="replay", manifest=manifest.directory)
    new_digest = _source.source_digest(sys.modules["ext_scale_mod"].ext_scale_kernel)
    assert new_digest != entry["source_digest"]
    assert entry["source_digest"] in str(e.value) and new_digest in str(e.value) and "stale" in str(e.value)
    with pytest.raises(TilefishManifestError, match="stale"):  # a pin vouches for the manifest, not for a source that is known to differ
        tilefish.run(ext_scale, x, mode="replay", manifest=manifest.directory, pin=manifest.digest)
    assert torch.equal(tilefish.run(ext_scale, x, mode="jit"), x * 3)
    again = runtime.export(ext_scale, tmp_path / "m", overwrite=True)  # exported again: the new binary
    assert torch.equal(tilefish.run(ext_scale, x, mode="replay", manifest=again.directory), x * 3)


def test_a_hand_written_kernel_that_cannot_be_recomputed_needs_a_pin(ext_dir, tmp_path, monkeypatch):
    x = randn((64,), torch.float32, 0)
    write_ext_module(ext_dir, "2.0")
    manifest = runtime.export(ext_scale, tmp_path / "m")
    monkeypatch.setattr(runtime, "_current_source_digest", lambda spec: None)
    with pytest.raises(TilefishManifestError, match="cannot tell whether the binary for 'ext_scale' .* was built from the current source.*cannot be recomputed"):
        tilefish.run(ext_scale, x, mode="replay", manifest=manifest.directory)
    pinned = tilefish.run(ext_scale, x, mode="replay", manifest=manifest.directory, pin=manifest.digest)
    assert torch.equal(pinned, x * 2)


def test_a_replay_without_triton_cannot_recompute_a_source_and_needs_a_pin(ext_dir, tmp_path):
    """In a process where Triton cannot be imported the kernel's module does not import either."""
    write_ext_module(ext_dir, "2.0")
    manifest = runtime.export(ext_scale, tmp_path / "m")
    child = (
        "import sys\n"
        "sys.modules['triton'] = None\n"
        "import torch, tilefish\n"
        "from tilefish import Dim, Out, Tensor, f32\n"
        "Count = Dim('Count', divisible_by=16)\n"
        "@tilefish.extern(triton='ext_scale_mod:ext_scale_kernel', num_warps=4, grid=lambda Count: (tilefish.cdiv(Count, 256),))\n"
        "def ext_scale(x_ptr: Tensor[Count, f32], y_ptr: Out[Tensor[Count, f32]], n_elements: Count, *, BLOCK: tilefish.const = 256): ...\n"
        "x = torch.arange(64, dtype=torch.float32, device='cuda')\n"
        "directory, pin = sys.argv[1], sys.argv[2]\n"
        "try:\n"
        "    tilefish.run(ext_scale, x, mode='replay', manifest=directory)\n"
        "    print('RAN WITHOUT A PIN')\n"
        "except tilefish.TilefishManifestError as e:\n"
        "    print('refused:', 'cannot be recomputed' in str(e))\n"
        "y = tilefish.run(ext_scale, x, mode='replay', manifest=directory, pin=pin)\n"
        "print('pinned replay correct:', bool((y == x * 2).all()))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(pathlib.Path(tilefish.__file__).resolve().parent.parent), str(ext_dir), env.get("PYTHONPATH", "")])
    r = subprocess.run([sys.executable, "-c", child, str(manifest.directory), manifest.digest], env=env, capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "refused: True" in r.stdout and "pinned replay correct: True" in r.stdout and "RAN WITHOUT A PIN" not in r.stdout


# -- the pin and what the directory may say ----------------------------------------------------------------------------------------------


def test_export_returns_the_digest_to_pin(tmp_path):
    x = randn((7, 1000), torch.float32, 0, scale=3.0)
    manifest = runtime.export(softmax, tmp_path)
    assert isinstance(manifest, Manifest) and manifest.directory == tmp_path and manifest.path == tmp_path / "manifest.json"
    assert manifest.digest == Manifest.load(tmp_path).digest and len(manifest.digest) == 64
    expected = tilefish.run(softmax, x, mode="jit")
    assert same_bytes(tilefish.run(softmax, x, mode="replay", manifest=tmp_path, pin=manifest.digest), expected)


def test_a_pin_refuses_a_directory_that_changed_in_a_way_the_digests_of_the_files_do_not_show(tmp_path):
    x = randn((7, 1000), torch.float32, 0)
    manifest = runtime.export(softmax, tmp_path)
    doc = json.loads((tmp_path / "manifest.json").read_text())
    doc["kernels"][0]["shared"] += 8  # a field no file digest covers
    (tmp_path / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match="it is not the manifest that was pinned"):
        tilefish.run(softmax, x, mode="replay", manifest=tmp_path, pin=manifest.digest)


def test_shared_zero_for_a_kernel_that_uses_shared_memory_is_refused_before_it_can_poison_the_context(tmp_path):
    x = randn((7, 1000), torch.float32, 0)
    runtime.export(softmax, tmp_path)
    doc = json.loads((tmp_path / "manifest.json").read_text())
    assert doc["kernels"][0]["shared"] > 0
    doc["kernels"][0]["shared"] = 0
    (tmp_path / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match=r"has a shared memory section .* shared = 0"):
        tilefish.run(softmax, x, mode="replay", manifest=tmp_path)
    torch.cuda.synchronize()  # the context is fine
    assert torch.isfinite(tilefish.run(softmax, x, mode="jit")).all()


def test_shared_memory_beyond_the_device_is_refused(tmp_path):
    x = randn((7, 1000), torch.float32, 0)
    runtime.export(softmax, tmp_path)
    doc = json.loads((tmp_path / "manifest.json").read_text())
    doc["kernels"][0]["shared"] = 1 << 22
    (tmp_path / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishRuntimeError, match=r"needs 0 bytes of static and 4194304 bytes of dynamic shared memory, more than the \d+ bytes a block may use"):
        tilefish.run(softmax, x, mode="replay", manifest=tmp_path)
    assert torch.isfinite(tilefish.run(softmax, x, mode="jit")).all()


def test_the_elf_of_every_triton_kernel_tells_whether_it_uses_shared_memory():
    """``.nv.shared.<symbol>`` is there exactly when Triton's metadata says the kernel needs dynamic shared memory."""
    from tilefish._elf import shared_sizes, uses_shared

    for arch in ("sm_80", "sm_90", "sm_100", "sm_103", "sm_120", ARCH):
        for kernel in (softmax, gelu_exact):
            low = kernel.lower()
            (spec,) = low.specs()
            b = build(spec, arch=arch, fn=runtime.load_lowered(low))
            elf = parse_elf(b.cubin)
            assert uses_shared(elf, b.symbol) == (b.shared > 0), (arch, kernel.name, b.shared)
            declared, reserved = shared_sizes(elf, b.symbol)
            assert declared in ((0, 1024) if b.shared else (0,)) and reserved in (0, 64)  # its size is not the signal: 0 on sm_80


# -- the architecture ------------------------------------------------------------------------------------------------------------------------


def test_replay_looks_up_the_binary_of_the_devices_architecture(tmp_path, monkeypatch):
    x = randn((7, 1000), torch.float32, 0)
    low = softmax.lower()
    runtime.export(low, tmp_path)  # this device's architecture
    runtime.export(low, tmp_path, arch="sm_90")  # merged into the same manifest
    m = Manifest.load(tmp_path)
    assert sorted(a.arch for a in m.artifacts) == sorted({ARCH, "sm_90"})
    seen = []

    class NoLaunch:
        def launch(self, grid, args, stream):
            seen.append(grid)

    def fake_load(built, index):
        seen.append(built.arch)
        return NoLaunch()

    monkeypatch.setattr(runtime._cuda, "load", fake_load)
    for capability, arch in ((90, "sm_90"), (CAP[0] * 10 + CAP[1], ARCH)):
        seen.clear()
        monkeypatch.setattr(runtime, "_capability", lambda torch_, index, c=capability: c)
        tilefish.run(softmax, x, mode="replay", manifest=tmp_path)
        assert seen[0] == arch  # the architecture of the (mocked) device, not a hardcoded one
    monkeypatch.setattr(runtime, "_capability", lambda torch_, index: 100)
    with pytest.raises(TilefishManifestError, match=rf"no binary for 'softmax' on sm_100 \(it has {', '.join(sorted({ARCH, 'sm_90'}))}\)"):
        tilefish.run(softmax, x, mode="replay", manifest=tmp_path)


def test_the_build_cache_is_per_architecture(monkeypatch):
    low = softmax.lower()
    (spec,) = low.specs()
    fn = runtime.load_lowered(low)
    runtime.clear_caches()
    here = runtime.build_for_device(spec, fn, 0)
    monkeypatch.setattr(runtime, "_capability", lambda torch_, index: 90)
    there = runtime.build_for_device(spec, fn, 0)
    assert here.arch == ARCH and there.arch == "sm_90" and here is not there
    assert runtime.build_for_device(spec, fn, 0) is there
    monkeypatch.undo()
    assert runtime.build_for_device(spec, fn, 0) is here


# -- export merges ---------------------------------------------------------------------------------------------------------------------------


def test_export_into_an_existing_directory_keeps_what_is_there(tmp_path):
    first = runtime.export(softmax, tmp_path)
    second = runtime.export(gelu_exact, tmp_path)  # another kernel, same directory
    third = runtime.export(softmax, tmp_path, arch="sm_90")  # another architecture
    m = Manifest.load(tmp_path)
    assert sorted((a.stem, a.arch) for a in m.artifacts) == sorted({("softmax", ARCH), ("gelu_exact", ARCH), ("softmax", "sm_90")})
    assert len({first.digest, second.digest, third.digest}) == 3 and third.digest == m.digest
    x = randn((7, 1000), torch.float32, 0)
    assert same_bytes(tilefish.run(softmax, x, mode="replay", manifest=tmp_path), tilefish.run(softmax, x, mode="jit"))
    assert tilefish.run(gelu_exact, randn((64,), BF16, 0), mode="replay", manifest=tmp_path).shape == (64,)


def test_export_does_not_replace_a_different_binary_unless_told_to(tmp_path):
    runtime.export(softmax, tmp_path)
    before = Manifest.load(tmp_path).digest
    other = softmax.lower(fp_fusion=False)  # the same stem, another declaration
    with pytest.raises(TilefishManifestError, match=r"already has a different binary for 'softmax' on .* \(sha256_code differs\); pass overwrite=True"):
        runtime.export(other, tmp_path)
    assert Manifest.load(tmp_path).digest == before  # nothing changed
    again = runtime.export(softmax, tmp_path)  # the same binary again is not a conflict
    assert len(again.artifacts) == 1
    replaced = runtime.export(other, tmp_path, overwrite=True)
    assert replaced.digest != before and replaced.get("softmax", ARCH).entry["options"]["enable_fp_fusion"] is False


# -- launching what does not fit --------------------------------------------------------------------------------------------------------------


def loaded_softmax():
    low = softmax.lower()
    (spec,) = low.specs()
    built = runtime.build_for_device(spec, runtime.load_lowered(low), 0)
    torch.zeros(1, device="cuda")
    return _cuda.load(built, 0), spec


def test_a_launch_that_does_not_fit_is_refused_and_the_context_survives():
    loaded, spec = loaded_softmax()
    x, out = torch.ones(1, 1000, device="cuda"), torch.zeros(1, 1000, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    good = [("pointer", x.data_ptr()), ("pointer", out.data_ptr())]
    assert [a.name for a in spec.args if a.kind != "constexpr"] == ["x", "out"]
    loaded.launch((1,), good, stream)
    torch.cuda.synchronize()
    assert float(out.sum()) == pytest.approx(1.0, rel=1e-5)  # a softmax row sums to one
    for grid, args, message in (
        ((1,), good[:1], r"takes 2 arguments \(x, out\), got 1"),  # a missing argument used to read past the array
        ((1,), [], r"got 0"),
        ((1,), good + [("scalar", 1)], r"got 3"),
        ((1,), [("scalar", 1), good[1]], r"is a pointer in the ABI"),
        ((0,), good, r"has 0 blocks along X"),
        ((2**31,), good, r"has 2147483648 blocks along X"),
        ((2**32 + 1,), good, r"along X"),
        ((1, 65536), good, r"has 65536 blocks along Y"),
        ((-1,), good, r"has -1 blocks"),
        ((1,), [("pointer", -1), good[1]], r"outside the 64-bit address range"),
    ):
        with pytest.raises(TilefishRuntimeError, match=message):
            loaded.launch(grid, args, stream)
    torch.cuda.synchronize()  # nothing was launched: the context is fine and the next launch works
    out.zero_()
    loaded.launch((1,), good, stream)
    torch.cuda.synchronize()
    assert float(out.sum()) == pytest.approx(1.0, rel=1e-5)


def test_the_devices_limits_are_what_the_driver_says():
    loaded, _ = loaded_softmax()
    assert loaded.grid_limits[0] == 2**31 - 1 and loaded.grid_limits[1] == loaded.grid_limits[2] == 65535


# -- the stream ---------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["jit", "replay"])
def test_the_launch_is_ordered_on_torchs_current_stream(mode, tmp_path):
    """The input is written on a side stream after a long wait; a launch on stream 0 would read it before that."""
    manifest = runtime.export(gelu_exact, tmp_path).directory if mode == "replay" else None
    kw = {"manifest": manifest} if manifest is not None else {}
    n = 1 << 16
    side = torch.cuda.Stream()
    x = torch.zeros(n, dtype=BF16, device="cuda")
    torch.cuda.synchronize()
    with torch.cuda.stream(side):
        torch.cuda._sleep(int(2e9))  # a long wait on this stream (about a second)
        x.fill_(5.0)  # the input only exists after it
        out = tilefish.run(gelu_exact, x, mode=mode, **kw)
        z = out.float() + 0.0
    side.synchronize()
    expected = tilefish.run(gelu_exact, torch.full((n,), 5.0, dtype=BF16, device="cuda"), mode="jit")
    assert same_bytes(out, expected), "the kernel ran before the input was written: it was not launched on the current stream"
    assert torch.equal(z, expected.float())


# -- checks of run() that nothing else trips -----------------------------------------------------------------------------------------------------


def extern_with(element_counts=None, params=None, grid=None, name="probe"):
    """An ``ExternOp`` made the way the lowering makes one (no check that the dimensions are derivable)."""
    from tilefish.decorators import ExternOp

    def stub():
        pass

    stub.__name__ = name
    return ExternOp(stub, name, Signature(params, ()), "no_such_module:no_such_kernel", 4, 3, grid, element_counts=element_counts)


def test_a_runtime_scalar_must_agree_with_the_element_count_it_stands_for_is_checked():
    M = Dim("M")
    params = [
        Param("x", 0, "tensor", "in", type=Tensor[M, 4, f32]),
        Param("y", 1, "tensor", "out", type=Tensor[M, 4, f32]),
        Param("n_elem", 2, "dim", dim=M),  # the declaration says M, the count of x is 4 * M
    ]
    ext = extern_with({"n_elem": "x"}, params, (as_dim(M),))
    with pytest.raises(TilefishPreconditionError, match=r"runtime argument 'n_elem' is M = 3, but the tensor 'x' has 12 elements") as e:
        tilefish.run(ext, torch.zeros(3, 4, device="cuda"), mode="jit")
    assert e.value.kind == "shape" and e.value.arg == "n_elem"


def test_a_runtime_scalar_that_does_not_fit_an_i32_is_checked():
    M = Dim("M")
    params = [
        Param("x", 0, "tensor", "in", type=Tensor[M, f32]),
        Param("y", 1, "tensor", "out", type=Tensor[M, f32]),
        Param("n_big", 2, "dim", dim=1_000_000 * M),
    ]
    ext = extern_with(None, params, (as_dim(M),))
    with pytest.raises(TilefishPreconditionError, match=r"runtime argument 'n_big' is 3000000000, which does not fit the 32-bit signed integer") as e:
        tilefish.run(ext, torch.zeros(3000, device="cuda"), mode="jit")
    assert e.value.arg == "n_big"


def test_a_constraint_on_a_dimension_found_from_an_element_count_is_checked():
    A, B, n = Dim("A"), Dim("B"), Dim("n", divisible_by=16)
    params = [
        Param("x", 0, "tensor", "in", type=Tensor[A, B, f32]),
        Param("y", 1, "tensor", "out", type=Tensor[A, B, f32]),
        Param("n_elem", 2, "dim", dim=n),
    ]
    ext = extern_with({"n_elem": "x"}, params, (cdiv(n, 256),))
    # the first check sees A and B, which are unconstrained; n only exists once it is taken from the element count
    with pytest.raises(TilefishPreconditionError, match=r"dimension n is 35, but it is declared divisible_by=16") as e:
        tilefish.run(ext, torch.zeros(5, 7, device="cuda"), mode="jit")
    assert e.value.kind == "divisible"


def test_the_binary_must_agree_with_the_spec_about_its_warps_is_checked(monkeypatch):
    x = randn((7, 1000), torch.float32, 0)
    real = runtime.build_for_device
    monkeypatch.setattr(runtime, "build_for_device", lambda spec, fn, index: dataclasses.replace(real(spec, fn, index), num_warps=8))
    with pytest.raises(TilefishRuntimeError, match="the binary of 'softmax' disagrees with its spec about num_warps"):
        tilefish.run(softmax, x, mode="jit")
    monkeypatch.setattr(
        runtime,
        "build_for_device",
        lambda spec, fn, index: dataclasses.replace(real(spec, fn, index), spec=dataclasses.replace(spec, block=(64, 1, 1))),
    )
    with pytest.raises(TilefishRuntimeError, match="disagrees with its spec about num_warps"):
        tilefish.run(softmax, x, mode="jit")


# -- build() checks what Triton did ----------------------------------------------------------------------------------------------------------


REAL_COMPILE = triton.compile


def lying_compile(change):
    """``triton.compile`` that returns the real result with its metadata changed by ``change``."""
    def compile_(*a, **k):
        compiled = REAL_COMPILE(*a, **k)
        compiled.metadata = change(compiled.metadata)
        return compiled

    return compile_


def test_build_checks_that_triton_applied_the_options_it_was_given(monkeypatch):
    low = softmax.lower()
    (spec,) = low.specs()
    fn = runtime.load_lowered(low)
    monkeypatch.setattr(triton, "compile", lying_compile(lambda md: md._replace(num_warps=2)))
    with pytest.raises(TilefishBuildError, match=r"the compile option num_warps=4 was not applied \(Triton used 2\)"):
        build(spec, arch=ARCH, fn=fn)
    monkeypatch.setattr(triton, "compile", lying_compile(lambda md: md._replace(enable_fp_fusion=False)))
    with pytest.raises(TilefishBuildError, match=r"the compile option enable_fp_fusion=True was not applied \(Triton used False\)"):
        build(spec, arch=ARCH, fn=fn)
    monkeypatch.setattr(triton, "compile", lying_compile(lambda md: md))
    assert build(spec, arch=ARCH, fn=fn).num_warps == 4


def test_build_checks_that_triton_built_for_the_architecture_it_was_asked_for(monkeypatch):
    from triton.backends.compiler import GPUTarget

    low = softmax.lower()
    (spec,) = low.specs()
    fn = runtime.load_lowered(low)
    monkeypatch.setattr(triton, "compile", lying_compile(lambda md: md._replace(target=GPUTarget("cuda", 80, 32))))
    with pytest.raises(TilefishBuildError, match=rf"asked for sm_{CAP[0] * 10 + CAP[1]}, Triton built for"):
        build(spec, arch=ARCH, fn=fn)


def test_build_checks_that_the_function_has_the_parameters_of_the_spec():
    low = softmax.lower()
    (spec,) = low.specs()
    other = runtime.load_lowered(gelu_exact.lower())
    with pytest.raises(TilefishBuildError, match=r"are not the ones its compile spec 'softmax' was made for"):
        build(spec, arch=ARCH, fn=other)


def test_an_extern_name_that_would_leave_the_directory_is_not_declared():
    with pytest.raises(tilefish.TilefishDeclarationError, match="name must be a plain identifier"):
        tilefish.extern(triton="ext_scale_mod:ext_scale_kernel", grid=lambda Count: (Count,), name="../outside")


def test_the_cubin_of_a_manifest_is_written_where_the_manifest_says(tmp_path):
    manifest = runtime.export(softmax, tmp_path / "out")
    entry = manifest.get("softmax", ARCH).entry
    assert hashlib.sha256((tmp_path / "out" / entry["cubin"]).read_bytes()).hexdigest() == entry["sha256"]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted([entry["cubin"], "manifest.json"])
    assert not [p for p in tmp_path.rglob("*.cubin") if p.parent != tmp_path / "out"]


def test_an_attacker_who_rewrites_the_cubin_and_every_digest_is_caught_only_by_a_pin(tmp_path):
    """The trust model: the digests in a directory vouch for nothing by themselves, the pin does."""
    from tilefish._elf import code_digest

    x = randn((7, 1000), torch.float32, 0)
    manifest = runtime.export(softmax, tmp_path)
    entry = json.loads((tmp_path / "manifest.json").read_text())["kernels"][0]
    path = tmp_path / entry["cubin"]
    blob = bytearray(path.read_bytes())
    text = parse_elf(bytes(blob)).section(f".text.{entry['symbol']}")
    blob[text.offset + 16] ^= 0x01  # one bit of machine code
    path.write_bytes(bytes(blob))
    doc = json.loads((tmp_path / "manifest.json").read_text())
    doc["kernels"][0]["sha256"] = hashlib.sha256(bytes(blob)).hexdigest()
    doc["kernels"][0]["sha256_code"] = code_digest(bytes(blob))
    (tmp_path / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match="it is not the manifest that was pinned"):
        tilefish.run(softmax, x, mode="replay", manifest=tmp_path, pin=manifest.digest)
    # without the pin the directory is trusted, as the documentation says: it replays the altered binary
    runtime.clear_caches()
    tilefish.run(softmax, x, mode="replay", manifest=tmp_path)


def test_a_declared_constraint_is_checked_before_a_binary_is_chosen():
    """Two violations at once: the declared ``divisible_by`` is reported first, not the missing bucket."""
    from tilefish.dims import next_pow2

    Mm, Sb = Dim("Mm", divisible_by=16), Dim("Sb", buckets=[32, 64])

    @tilefish.extern(triton="no_such_module:no_such_kernel", grid=lambda Mm: (Mm,))
    def two_constraints(x: Tensor[Mm, Sb, f32], y: Out[Tensor[Mm, Sb, f32]], *, BLOCK: tilefish.const = next_pow2(Sb)): ...

    with pytest.raises(TilefishPreconditionError, match=r"dimension Mm is 20, but it is declared divisible_by=16") as e:
        tilefish.run(two_constraints, torch.zeros(20, 40, device="cuda"), mode="jit")  # 40 is no bucket either
    assert e.value.kind == "divisible"
    with pytest.raises(TilefishPreconditionError, match="no padding") as e:
        tilefish.run(two_constraints, torch.zeros(32, 40, device="cuda"), mode="jit")
    assert e.value.kind == "bucket"


# -- H1: what Triton's cache_key does not see ------------------------------------------------------------------------------------------------

HK_FILES = {
    "hk_kernel": (
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "import hk_cfg\n"
        "import hk_helper_a\n"
        "\n"
        "\n"
        "@triton.jit\n"
        "def hk_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):\n"
        "    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
        "    mask = offsets < n_elements\n"
        "    x = tl.load(x_ptr + offsets, mask=mask)\n"
        "    y = hk_helper_a.apply(x.to(hk_cfg.DT)) * hk_cfg.SCALE\n"
        "    tl.store(y_ptr + offsets, y, mask=mask)\n"
    ),
    "hk_cfg": "import triton.language as tl\n\nSCALE = tl.constexpr(@SCALE@)\nDT = tl.@DT@\n",
    "hk_helper_a": "import triton\nimport triton.language as tl\n\nimport hk_helper_b\n\n\n@triton.jit\ndef apply(x):\n    return x + hk_helper_b.OFFSET\n",
    "hk_helper_b": "import triton.language as tl\n\nOFFSET = tl.constexpr(@OFFSET@)\n",
}
HK_BASE = {"SCALE": "2.0", "DT": "float32", "OFFSET": "0.0"}
HK_MODULES = tuple(HK_FILES)

HkN = Dim("HkN", divisible_by=16)


@tilefish.extern(triton="hk_kernel:hk_kernel", num_warps=4, grid=lambda HkN: (cdiv(HkN, 256),))
def hk(x_ptr: Tensor[HkN, f32], y_ptr: Out[Tensor[HkN, f32]], n_elements: HkN, *, BLOCK: tilefish.const = 256): ...


class HkProject:
    """Four modules: the kernel's, ``cfg`` (a constexpr and a dtype alias), and a helper that reads a constexpr of a helper below it."""

    def __init__(self, directory):
        self.directory = directory

    def write(self, **values):
        values = {**HK_BASE, **values}
        for name, text in HK_FILES.items():
            for key, value in values.items():
                text = text.replace(f"@{key}@", value)
            (self.directory / f"{name}.py").write_text(text)

    def load(self):
        """Import the modules from their files as they are now, as a new process would."""
        for name in HK_MODULES:
            sys.modules.pop(name, None)
        importlib.invalidate_caches()
        linecache.clearcache()
        return importlib.import_module("hk_kernel")


@pytest.fixture
def hkp(tmp_path, monkeypatch):
    directory = tmp_path / "hk"
    directory.mkdir()
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)  # an edit of one byte within a second must not meet a stale .pyc
    yield HkProject(directory)
    for name in HK_MODULES:
        sys.modules.pop(name, None)


def hk_expected(x, values):
    """What the kernel computes: ``(x.to(DT) + OFFSET) * SCALE``, with the roundings of its dtype."""
    values = {**HK_BASE, **values}
    t = {"float32": torch.float32, "bfloat16": torch.bfloat16}[values["DT"]]
    return ((x.to(t) + float(values["OFFSET"])) * float(values["SCALE"])).float()


HK_EDITS = [
    ("a constexpr read as a module attribute (cfg.SCALE)", {"SCALE": "3.0"}),
    ("a dtype alias (cfg.DT)", {"DT": "bfloat16"}),
    ("a constant in a helper module two levels away", {"OFFSET": "1.0"}),
]


@pytest.mark.parametrize("label, edit", HK_EDITS, ids=[label for label, _ in HK_EDITS])
def test_an_edit_in_a_module_the_kernel_reaches_is_not_replayed_stale_with_or_without_a_pin(hkp, tmp_path, label, edit):
    """What Triton's ``cache_key`` misses: it is the same before and after these edits, so its disk cache would serve the old cubin too."""
    x = 1 + torch.arange(64, dtype=torch.float32, device="cuda") / 1024  # exact in f32, not in bf16
    hkp.write()
    hkp.load()
    old = hk_expected(x, {})
    manifest = runtime.export(hk, tmp_path / "m")
    assert torch.equal(tilefish.run(hk, x, mode="replay", manifest=manifest.directory), old)
    assert torch.equal(tilefish.run(hk, x, mode="replay", manifest=manifest.directory, pin=manifest.digest), old)
    assert torch.equal(tilefish.run(hk, x, mode="jit"), old)
    entry = manifest.get("hk", ARCH).entry
    assert sorted(_source.reachable(sys.modules["hk_kernel"].hk_kernel).files) == sorted(HK_MODULES)
    new = hk_expected(x, edit)
    assert not torch.equal(old, new)  # the edit changes the result: serving the old binary is visible

    hkp.write(**edit)
    hkp.load()
    new_digest = _source.source_digest(sys.modules["hk_kernel"].hk_kernel)
    assert new_digest != entry["source_digest"]
    for pin in (None, manifest.digest):  # refused with and without the pin of the old manifest
        with pytest.raises(TilefishManifestError) as e:
            tilefish.run(hk, x, mode="replay", manifest=manifest.directory, pin=pin)
        assert entry["source_digest"] in str(e.value) and new_digest in str(e.value) and "stale" in str(e.value) and "export it again" in str(e.value)
    assert torch.equal(tilefish.run(hk, x, mode="jit"), new)  # not the old binary, from the build cache or from Triton's disk cache
    again = runtime.export(hk, tmp_path / "m", overwrite=True)
    assert again.digest != manifest.digest and again.get("hk", ARCH).entry["source_digest"] == new_digest
    assert torch.equal(tilefish.run(hk, x, mode="replay", manifest=again.directory, pin=again.digest), new)
    assert same_bytes(tilefish.run(hk, x, mode="replay", manifest=again.directory), tilefish.run(hk, x, mode="jit"))


def test_the_old_exports_binary_is_not_what_a_new_export_of_the_edited_kernel_writes(hkp, tmp_path):
    """``export`` is stale too when Triton's cache serves the cubin: two exports from two sources must differ in machine code."""
    hkp.write()
    hkp.load()
    first = runtime.export(hk, tmp_path / "one")
    hkp.write(SCALE="3.0")
    hkp.load()
    second = runtime.export(hk, tmp_path / "two")
    assert first.get("hk", ARCH).entry["sha256_code"] != second.get("hk", ARCH).entry["sha256_code"]


def test_a_file_edited_after_the_process_imported_it_is_not_built(hkp, tmp_path):
    """The digest is of the files on disk, so a build refuses a function the process holds in an older version."""
    x = 1 + torch.arange(64, dtype=torch.float32, device="cuda") / 1024
    hkp.write()
    hkp.load()
    runtime.export(hk, tmp_path / "m")
    source = hkp.directory / "hk_kernel.py"
    source.write_text(source.read_text().replace("* hk_cfg.SCALE", "* hk_cfg.SCALE * 1.0"))  # edited, not reloaded
    runtime.clear_caches()
    with pytest.raises(TilefishBuildError, match=r"'hk_kernel' was imported from an older version of '.*hk_kernel.py'.*Restart the process, or reload"):
        runtime.export(hk, tmp_path / "again", overwrite=True)
    with pytest.raises(TilefishBuildError, match="older version"):
        tilefish.run(hk, x, mode="jit")
    hkp.load()  # reloaded: fine
    assert torch.equal(tilefish.run(hk, x, mode="jit"), hk_expected(x, {}))
    helper = hkp.directory / "hk_helper_a.py"
    helper.write_text(helper.read_text() + "\n")  # a helper's function is checked too: appending changes no line of it ...
    runtime.clear_caches()
    assert torch.equal(tilefish.run(hk, x, mode="jit"), hk_expected(x, {}))
    helper.write_text("# moved\n" + helper.read_text())  # ... but a line above it moves it
    runtime.clear_caches()
    with pytest.raises(TilefishBuildError, match=r"'apply' was imported from an older version"):
        tilefish.run(hk, x, mode="jit")


# -- H2: build() stamps a source digest on what it compiled ---------------------------------------------------------------------------------------


def test_a_stale_generated_file_on_sys_path_is_not_built_as_the_new_kernel(tmp_path, monkeypatch):
    old, new = scale_kernel(2.0), scale_kernel(3.0)  # the same name and signature, another body
    src = tmp_path / "src"
    low_old, low_new = old.lower(), new.lower()
    assert low_old.module == low_new.module and low_old.source_sha256 != low_new.source_sha256
    path = low_old.write(src)  # what an earlier run left behind
    monkeypatch.syspath_prepend(str(src))
    sys.modules.pop(low_new.module, None)
    (spec,) = low_new.specs()
    with pytest.raises(TilefishBuildError) as e:
        build(spec, arch=ARCH)  # fn=None: imports `module:fn` from sys.path
    message = str(e.value)
    assert str(path) in message and low_old.source_sha256 in message and low_new.source_sha256 in message and "stale" in message
    with pytest.raises(TilefishBuildError, match="the file is not the source this kernel generates now"):
        runtime.export(low_new.extern, tmp_path / "d")  # the same path through export
    assert not (tmp_path / "d" / "manifest.json").exists()  # nothing was written
    # the matching file builds
    low_new.write(src)
    sys.modules.pop(low_new.module, None)
    importlib.invalidate_caches()
    linecache.clearcache()
    built = build(spec, arch=ARCH)
    assert built.source_digest is None and len(built.cubin) > 0  # a generated kernel's spec carries its source; the entry does not need a digest


def test_another_function_passed_as_fn_is_not_built_as_the_spec_s_kernel(tmp_path):
    old, new = scale_kernel(2.0), scale_kernel(3.0)
    (spec,) = new.lower().specs()
    stale = runtime.load_lowered(old.lower())  # a JITFunction with the very parameters of the spec, and another body
    with pytest.raises(TilefishBuildError, match=r"whose sha256 is .*, but the compile spec 'scale_k' is for Triton source with sha256 " + spec.source_sha256):
        build(spec, arch=ARCH, fn=stale)
    assert build(spec, arch=ARCH, fn=runtime.load_lowered(new.lower())).symbol


def test_a_generated_file_edited_after_it_was_imported_is_not_built(tmp_path, monkeypatch):
    """The file now has the spec's source, but the module in ``sys.modules`` is the old one."""
    old, new = scale_kernel(2.0), scale_kernel(3.0)
    src = tmp_path / "src"
    low_old, low_new = old.lower(), new.lower()
    low_old.write(src)
    monkeypatch.syspath_prepend(str(src))
    sys.modules.pop(low_new.module, None)
    importlib.invalidate_caches()
    imported = importlib.import_module(low_new.module)  # the old source
    low_new.write(src)  # the file is overwritten with the new one, the process keeps the old function
    linecache.clearcache()
    (spec,) = low_new.specs()
    with pytest.raises(TilefishBuildError, match=r"was imported from an older version of .*Restart the process, or reload"):
        build(spec, arch=ARCH, fn=getattr(imported, low_new.fn_name))


def test_build_compares_the_source_digest_it_is_given(ext_dir):
    write_ext_module(ext_dir, "2.0")
    (spec,) = compile_specs_of(ext_scale)
    digest = _source.source_digest(sys.modules["ext_scale_mod"].ext_scale_kernel)
    assert build(spec, arch=ARCH).source_digest == build(spec, arch=ARCH, source_digest=digest).source_digest == digest
    with pytest.raises(TilefishBuildError, match=rf"has digest {digest}, but {'0' * 64} was expected"):
        build(spec, arch=ARCH, source_digest="0" * 64)


def compile_specs_of(op):
    from tilefish.aot import compile_specs

    return compile_specs(op)


def test_an_entry_of_a_generated_kernel_needs_no_source_digest_and_one_of_an_extern_has_one(ext_dir, tmp_path):
    write_ext_module(ext_dir, "2.0")
    generated = runtime.export(softmax, tmp_path / "g").get("softmax", ARCH).entry
    assert "source_sha256" in generated and "source_digest" not in generated and "source_digest_kind" not in generated
    handwritten = runtime.export(ext_scale, tmp_path / "e").get("ext_scale", ARCH).entry
    assert "source_digest" in handwritten and "source_sha256" not in handwritten


def test_a_generated_kernels_build_key_is_its_spec_and_an_externs_is_its_files(ext_dir, monkeypatch):
    """A generated kernel carries its source in its spec, so a call does not read any file; a hand-written one is keyed by its digest."""
    x = randn((7, 1000), torch.float32, 0)
    y = randn((64,), torch.float32, 0)
    write_ext_module(ext_dir, "2.0")
    runtime.clear_caches()

    def boom(fn):
        raise AssertionError("a file was read for the key of a generated kernel")

    monkeypatch.setattr(_source, "source_digest", boom)
    assert torch.isfinite(tilefish.run(softmax, x, mode="jit")).all()  # builds, and runs again from the cache, without a digest
    assert torch.isfinite(tilefish.run(softmax, x, mode="jit")).all()
    with pytest.raises(AssertionError, match="a file was read"):
        tilefish.run(ext_scale, y, mode="jit")
    monkeypatch.undo()
    assert torch.equal(tilefish.run(ext_scale, y, mode="jit"), y * 2)


# -- no build reads Triton's disk cache ---------------------------------------------------------------------------------------------------------


def test_every_build_compiles_from_scratch_and_puts_the_knob_back(monkeypatch):
    low = softmax.lower()
    (spec,) = low.specs()
    fn = runtime.load_lowered(low)
    knob = triton.knobs.compilation
    seen = []

    def spy(*a, **k):
        seen.append(knob.always_compile)
        return REAL_COMPILE(*a, **k)

    monkeypatch.setattr(triton, "compile", spy)
    assert knob.always_compile is False
    build(spec, arch=ARCH, fn=fn)
    assert seen == [True] and knob.always_compile is False
    knob.always_compile = True  # a caller that set it itself finds it as it was
    try:
        build(spec, arch=ARCH, fn=fn)
        assert seen == [True, True] and knob.always_compile is True
    finally:
        knob.always_compile = False

    def boom(*a, **k):
        seen.append(knob.always_compile)
        raise RuntimeError("compiler exploded")

    monkeypatch.setattr(triton, "compile", boom)
    with pytest.raises(TilefishBuildError, match="compiler exploded"):
        build(spec, arch=ARCH, fn=fn)
    assert seen[-1] is True and knob.always_compile is False  # restored when the compile fails too


def test_a_build_does_not_reuse_the_binary_in_triton_s_disk_cache(monkeypatch, tmp_path):
    """The same cache key, other content: a cache that was written for the old kernel must not be served for the new one."""
    monkeypatch.setattr(triton.knobs.cache, "dir", str(tmp_path / "cache"))
    low = softmax.lower()
    (spec,) = low.specs()
    fn = runtime.load_lowered(low)
    first = build(spec, arch=ARCH, fn=fn)
    cached = list((tmp_path / "cache").rglob("*.cubin"))
    assert cached  # Triton wrote it
    for p in cached:
        p.write_bytes(b"not a cubin: poisoned")  # what a stale or damaged entry looks like
    second = build(spec, arch=ARCH, fn=fn)  # a cache hit would hand back the poisoned file
    assert second.sha256_code == first.sha256_code and second.cubin[:4] == b"\x7fELF"


# -- export checks the directory it wrote against the builds in memory ------------------------------------------------------------------------------------


def test_export_returns_the_digest_of_what_it_built_and_checks_the_directory_against_it(tmp_path, monkeypatch):
    runtime.export(softmax, tmp_path)
    merged = runtime.export(gelu_exact, tmp_path)  # merged with an entry that was read from the directory
    assert merged.digest == Manifest.load(tmp_path).digest
    monkeypatch.setattr(runtime, "manifest_digest", lambda doc: "0" * 64)  # what the in-memory document would have said, if it differed
    with pytest.raises(TilefishManifestError, match=r"has digest [0-9a-f]{64} when it is read back, but the manifest that was written has digest " + "0" * 64):
        runtime.export(gelu_exact, tmp_path)


