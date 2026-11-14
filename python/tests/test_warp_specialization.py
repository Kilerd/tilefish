"""Warp-specialised kernels, with no Triton and no GPU: the build, the cache, the manifest and its readers, a launch, and the Rust emitter.

A kernel is warp-specialised when its TTGIR has ``ttg.warp_specialize`` regions: Gluon's ``gl.warp_specialize`` (a default partition, and worker
partitions with warp counts of their own: a TMA producer, an MMA issuer), or Triton's own warp specialisation of a ``tl.range(...,
warp_specialize=True)`` loop on sm_100 and up. Its partitions add warps to the ``num_warps`` it was compiled with, which are its default partition's,
and Triton 3.5.1 reports the total as the kernel's ``num_warps`` (``ttg.total-num-warps``: 4 + 1 + 1 is 8, 8 + 1 + 1 is 12). Held here:

* the build takes that total as the warps of a block (``BuiltKernel.num_warps``, ``block`` = ``32 *`` it) and records it in the manifest entry
  (``total_num_warps``; ``num_warps`` and ``options.num_warps`` stay the declared ones); a kernel that is not warp-specialised must still have exactly
  the declared warps, and the PTX's ``.reqntid`` must be the block;
* the build cache keeps and checks it; both readers refuse a total that is not more than ``num_warps`` or more than 32, a block that is not the
  total's, and the field on a prebuilt binary (``crates/tilefish/tests/warp_specialization.rs`` has the Rust reader's refusals);
* a replay launches with the compiled block, and refuses a binary whose warps are not the spec's (or more, for a warp-specialised one);
* the Rust emitter says the block of each architecture.

``fake_compile`` stands in for the frontends and the compiler; ``test_gpu_warp_specialization.py`` runs real warp-specialised kernels.
"""

import dataclasses
import importlib
import json
import linecache
import sys
import types
import warnings

import fake_compile
import pytest

import tilefish
from tilefish import Dim, Manifest, Out, Tensor, TilefishBuildError, TilefishCacheWarning, TilefishManifestError, TilefishRuntimeError, _cache, const, f16
from tilefish._build import check_launch_config, ptx_reqntid, warps_problem
from tilefish.aot import build, compile_specs, write_manifest

R, C = Dim("R"), Dim("C")
ARCH = 103

KERNELS = '''from fake_jit import gluon_jit, jit


@gluon_jit
def ws_copy(x, y, n_cols, BM):
    return (x, y, n_cols, BM)


@jit
def ws_loop(x, y, n_cols, BM):
    return (x, y, n_cols, BM)
'''


def declare(ref="ws_kernels:ws_copy", *, kind="gluon", num_warps=4, name="ws_copy"):
    """A tiled copy whose kernel the fake compiler warp-specialises when told to (a Gluon kernel, or a Triton one with ``kind="triton"``)."""

    @tilefish.extern(**{kind: ref}, num_warps=num_warps, grid=lambda R: (tilefish.cdiv(R, 64),), name=name)
    def op(x: Tensor[R, C, f16], y: Out[Tensor[R, C, f16]], n_cols: C, BM: const = 64): ...

    return op


@pytest.fixture
def world(tmp_path, monkeypatch):
    compiler = fake_compile.install(monkeypatch)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    directory = tmp_path / "project"
    directory.mkdir()
    (directory / "ws_kernels.py").write_text(KERNELS)
    monkeypatch.syspath_prepend(str(directory))
    sys.modules.pop("ws_kernels", None)
    importlib.invalidate_caches()
    linecache.clearcache()
    yield directory, compiler
    sys.modules.pop("ws_kernels", None)


def warp_specialized(compiler, total=8):
    """Make the fake compiler warp-specialise what it compiles, with ``total`` warps in all, and a PTX that requires that block."""
    compiler.total_warps = total
    compiler.reqntid = "block"


def export(tmp_path, built, name="out"):
    write_manifest(built if isinstance(built, list) else [built], tmp_path / name / "manifest.json", cubins=True)
    return tmp_path / name


# -- the build ----------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind, ref", [("gluon", "ws_kernels:ws_copy"), ("triton", "ws_kernels:ws_loop")])
def test_a_warp_specialised_kernel_is_built_with_the_warps_of_all_its_partitions(world, tmp_path, kind, ref):
    directory, compiler = world
    warp_specialized(compiler, 8)  # a 4-warp default partition, a 1-warp TMA producer and a 1-warp MMA issuer: 8 in Triton 3.5.1
    (spec,) = compile_specs(declare(ref, kind=kind))
    built = build(spec, arch=ARCH)
    assert (built.num_warps, built.warp_specialized, built.block) == (8, True, (256, 1, 1))
    assert spec.num_warps == 4 and spec.block == (128, 1, 1) and built.options["num_warps"] == 4  # the declaration and its compile option
    out = export(tmp_path, built)
    entry = json.loads((out / "manifest.json").read_text())["kernels"][0]
    assert (entry["num_warps"], entry["options"]["num_warps"], entry["total_num_warps"], entry["block"]) == (4, 4, 8, [256, 1, 1])
    replayed = Manifest.load(out).built_kernel(spec, f"sm_{ARCH}", pinned=True)
    assert (replayed.num_warps, replayed.warp_specialized, replayed.block) == (8, True, (256, 1, 1))
    check_launch_config(replayed, spec)


def test_a_wider_default_partition_has_its_warps_and_the_workers(world, tmp_path):
    directory, compiler = world
    warp_specialized(compiler, 12)  # an 8-warp default partition and two 1-warp workers, padded to a warp group: 12
    (spec,) = compile_specs(declare(num_warps=8))
    built = build(spec, arch=ARCH)
    assert (built.num_warps, built.block) == (12, (384, 1, 1))
    entry = json.loads((export(tmp_path, built) / "manifest.json").read_text())["kernels"][0]
    assert (entry["num_warps"], entry["total_num_warps"], entry["block"]) == (8, 12, [384, 1, 1])


def test_a_kernel_that_is_not_warp_specialised_has_exactly_the_declared_warps(world, tmp_path):
    directory, compiler = world
    (spec,) = compile_specs(declare())
    for silent in (8, 2):  # Triton reports other warps for a kernel with no warp-specialised region: a silent change, refused
        compiler.metadata_warps = silent
        with pytest.raises(TilefishBuildError, match=rf"the compile option num_warps=4 was not applied \(Triton used {silent}\)"):
            build(spec, arch=ARCH, cache=False)
    compiler.metadata_warps = None
    compiler.reqntid = "block"
    built = build(spec, arch=ARCH)
    assert (built.num_warps, built.warp_specialized, built.block) == (4, False, (128, 1, 1))
    entry = json.loads((export(tmp_path, built) / "manifest.json").read_text())["kernels"][0]
    assert "total_num_warps" not in entry and entry["block"] == [128, 1, 1] and entry["num_warps"] == 4  # an entry as it always was


@pytest.mark.parametrize("total, why", [(4, "add warps to the 4 of its default partition"), (2, "with 2 warps in all"), (64, "at most 32")])
def test_the_partitions_of_a_warp_specialised_kernel_add_warps_and_a_block_has_at_most_32(world, total, why):
    directory, compiler = world
    warp_specialized(compiler, total)
    (spec,) = compile_specs(declare())
    with pytest.raises(TilefishBuildError, match=why):
        build(spec, arch=ARCH, cache=False)


def test_the_threads_the_ptx_requires_are_the_block(world):
    directory, compiler = world
    (spec,) = compile_specs(declare())
    warp_specialized(compiler, 8)
    compiler.reqntid = 128  # the code was compiled for the declared warps, but the metadata says 8
    with pytest.raises(TilefishBuildError, match=r"requires 128 threads per block \(.reqntid\), but Triton's metadata says 8 warps \(256 threads\)"):
        build(spec, arch=ARCH, cache=False)
    compiler.total_warps = None
    compiler.reqntid = 256
    with pytest.raises(TilefishBuildError, match=r"requires 256 threads per block"):
        build(spec, arch=ARCH, cache=False)


def test_the_reqntid_of_a_ptx_entry_is_read_from_its_directives():
    sig = {"x": "*fp16", "n": "i32", "B": "constexpr"}
    assert ptx_reqntid(fake_compile.ptx_of(sig, 103, 256), "tf_k") == 256
    assert ptx_reqntid(fake_compile.ptx_of(sig, 103, "128, 2, 1"), "tf_k") == 256
    assert ptx_reqntid(fake_compile.ptx_of(sig, 103), "tf_k") is None
    assert ptx_reqntid(fake_compile.ptx_of(sig, 103, 256), "another") is None
    body = fake_compile.ptx_of(sig, 103) + "\t// .reqntid 64 in a comment of the body\n}\n"
    assert ptx_reqntid(body, "tf_k") is None  # only the directives between the parameters and the body


def test_the_warps_of_a_binary_are_its_declarations_or_more_when_warp_specialised():
    assert warps_problem(4, 4, False) is None and warps_problem(4, 8, True) is None and warps_problem(4, 32, True) is None
    assert "it has 8 warps, but the declaration gives num_warps=4" in warps_problem(4, 8, False)
    assert "add warps" in warps_problem(4, 4, True) and "at most 32" in warps_problem(4, 33, True)
    assert warps_problem(0, 0, False) is None and "prebuilt binary" in warps_problem(0, 8, True)
    assert "not an int" in warps_problem(4, True, False)


# -- the build cache ------------------------------------------------------------------------------------------------------------------------------


def test_the_cache_keeps_a_warp_specialised_build_and_checks_its_warps(world):
    directory, compiler = world
    warp_specialized(compiler, 8)
    (spec,) = compile_specs(declare())
    first = build(spec, arch=ARCH)
    hit = build(spec, arch=ARCH)
    assert compiler.count == 1 and (hit.num_warps, hit.warp_specialized, hit.block) == (8, True, (256, 1, 1))
    (path,) = list(_cache.default_cache().directory.rglob("*.json"))
    good = json.loads(path.read_text())
    assert (good["num_warps"], good["warp_specialized"]) == (8, True)
    for change, why in (
        ({"warp_specialized": False}, "it has 8 warps, but the declaration gives num_warps=4"),  # a silent change, as far as the entry says
        ({"num_warps": 4}, "add warps to the 4 of its default partition"),
        ({"num_warps": 16}, "requires 256 threads per block"),  # what the PTX of the entry requires is 8 warps
        ({"warp_specialized": "yes"}, "'warp_specialized' is not a bool"),
    ):
        path.write_text(json.dumps({**good, **change}))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            again = build(spec, arch=ARCH)
        assert any(why in str(w.message) for w in caught if w.category is TilefishCacheWarning), (change, [str(w.message) for w in caught])
        assert (again.num_warps, again.warp_specialized) == (8, True) and again.sha256_code == first.sha256_code


# -- the manifest readers ----------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def ws_manifest(world, tmp_path):
    directory, compiler = world
    warp_specialized(compiler, 8)
    (spec,) = compile_specs(declare())
    return spec, export(tmp_path, build(spec, arch=ARCH))


def edited(out, change):
    path = out / "manifest.json"
    doc = json.loads(path.read_text())
    change(doc["kernels"][0])
    path.write_text(json.dumps(doc))


@pytest.mark.parametrize(
    "change, why",
    [
        (lambda e: e.update(total_num_warps=4), r"total_num_warps must be an int above num_warps \(4"),
        (lambda e: e.update(total_num_warps=2), "total_num_warps must be an int above num_warps"),
        (lambda e: e.update(total_num_warps=33, block=[1056, 1, 1]), "and at most 32, got 33"),
        (lambda e: e.update(total_num_warps="8"), "total_num_warps must be an int"),
        (lambda e: e.update(total_num_warps=8.0), "total_num_warps must be an int"),
        (lambda e: e.update(total_num_warps=True), "total_num_warps must be an int"),
        (lambda e: e.update(block=[128, 1, 1]), r"is \[32 \* total_num_warps, 1, 1\] = \[256, 1, 1\], got \[128, 1, 1\]"),
        (lambda e: e.update(total_num_warps=16), r"= \[512, 1, 1\], got \[256, 1, 1\]"),
    ],
)
def test_the_reader_refuses_warps_that_are_not_those_of_a_warp_specialised_kernel(ws_manifest, change, why):
    spec, out = ws_manifest
    edited(out, change)
    with pytest.raises(TilefishManifestError, match=why):
        Manifest.load(out)


def test_a_block_of_other_warps_without_total_num_warps_is_not_what_the_spec_asks_for(ws_manifest):
    spec, out = ws_manifest
    edited(out, lambda e: e.update(total_num_warps=None))  # a JSON null is as absent: then the block must be the declared warps'
    with pytest.raises(TilefishManifestError, match=r"first difference at block\[0\]: the manifest has 256, the spec gives 128"):
        Manifest.load(out).built_kernel(spec, f"sm_{ARCH}", pinned=True)


def test_a_prebuilt_binary_has_no_total_num_warps(tmp_path):
    from tilefish._manifest import _check_warps

    with pytest.raises(TilefishManifestError, match="a prebuilt binary has no total_num_warps"):
        _check_warps("'scale' (sm_103)", {"total_num_warps": 4, "block": [128, 1, 1]}, "binary")
    _check_warps("'scale' (sm_103)", {"block": [64, 1, 1]}, "binary")


def test_the_compiled_warps_are_in_the_manifest_digest(world, tmp_path):
    directory, compiler = world
    (spec,) = compile_specs(declare())
    warp_specialized(compiler, 8)
    eight = Manifest.load(export(tmp_path, build(spec, arch=ARCH, cache=False), "eight")).digest
    warp_specialized(compiler, 12)
    twelve = Manifest.load(export(tmp_path, build(spec, arch=ARCH, cache=False), "twelve")).digest
    assert eight != twelve


def test_a_merge_does_not_replace_a_binary_of_other_warps(world, tmp_path):
    directory, compiler = world
    (spec,) = compile_specs(declare())
    warp_specialized(compiler, 8)
    out = export(tmp_path, build(spec, arch=ARCH, cache=False))
    warp_specialized(compiler, 12)
    with pytest.raises(TilefishManifestError, match="differs"):
        write_manifest([build(spec, arch=ARCH, cache=False)], out / "manifest.json", cubins=True, merge=True)


# -- a launch ---------------------------------------------------------------------------------------------------------------------------------------


class FakeDriver:
    """Just enough of ``cuda.bindings.driver`` for ``LoadedKernel`` to load a function and launch it; it records what ``cuLaunchKernel`` is given."""

    OK = 0

    def __init__(self):
        self.CUresult = types.SimpleNamespace(CUDA_SUCCESS=self.OK)
        names = ("MAX_GRID_DIM_X", "MAX_GRID_DIM_Y", "MAX_GRID_DIM_Z", "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN")
        self.CUdevice_attribute = types.SimpleNamespace(**{f"CU_DEVICE_ATTRIBUTE_{n}": n for n in names})
        self.CUfunction_attribute = types.SimpleNamespace(
            CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES="static", CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES="dynamic"
        )
        self.launches = []

    def cuCtxGetCurrent(self):
        return (self.OK, 1)

    def cuCtxGetDevice(self):
        return (self.OK, 0)

    def cuDeviceGetAttribute(self, attr, device):
        return (self.OK, 232448 if attr == "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN" else 65535)

    def cuModuleLoadData(self, cubin):
        return (self.OK, "module")

    def cuModuleGetFunction(self, module, name):
        return (self.OK, "function")

    def cuFuncGetAttribute(self, attr, function):
        return (self.OK, 0)

    def cuFuncSetAttribute(self, function, attr, value):
        return (self.OK,)

    def CUstream(self, stream):
        return stream

    def cuLaunchKernel(self, function, gx, gy, gz, bx, by, bz, shared, stream, params, extra):
        self.launches.append(((gx, gy, gz), (bx, by, bz), shared))
        return (self.OK,)


def test_a_launch_takes_the_block_of_the_compiled_kernel(world, monkeypatch):
    from tilefish import _cuda

    directory, compiler = world
    driver = FakeDriver()
    monkeypatch.setattr(_cuda, "_driver", lambda: driver)
    args = [("pointer", 0x7F0000000000), ("pointer", 0x7F0000100000), ("scalar", 256)]
    (spec,) = compile_specs(declare())
    plain = build(spec, arch=ARCH, cache=False)
    warp_specialized(compiler, 8)
    ws = build(spec, arch=ARCH, cache=False)
    assert _cuda.load_key(plain, 0) != _cuda.load_key(ws, 0)  # another block is another launch
    for built in (plain, ws):
        _cuda.LoadedKernel(built, 0).launch((4,), args, stream=0)
    assert [(grid, block) for grid, block, _ in driver.launches] == [((4, 1, 1), (128, 1, 1)), ((4, 1, 1), (256, 1, 1))]


def test_a_binary_whose_warps_are_not_the_specs_is_refused_at_a_launch(world):
    directory, compiler = world
    warp_specialized(compiler, 8)
    (spec,) = compile_specs(declare())
    ws = build(spec, arch=ARCH)
    check_launch_config(ws, spec)
    with pytest.raises(TilefishRuntimeError, match="disagrees with its spec about num_warps: it has 8 warps, but the declaration gives num_warps=4"):
        check_launch_config(dataclasses.replace(ws, warp_specialized=False), spec)
    (wider,) = compile_specs(declare(num_warps=8))
    with pytest.raises(TilefishRuntimeError, match="disagrees with its spec about num_warps"):
        check_launch_config(ws, wider)  # a binary of a default partition of 4 warps is not one of 8


# -- the Rust emitter -----------------------------------------------------------------------------------------------------------------------------


def test_the_rust_emitter_says_the_block_of_each_architecture(world, tmp_path):
    from tilefish.hosts import rust

    directory, compiler = world
    op = declare()
    (spec,) = compile_specs(op)
    plain = build(spec, arch=90)  # (Triton warp-specialises a loop on sm_100 and up only)
    warp_specialized(compiler, 8)
    ws = build(spec, arch=ARCH)
    out = export(tmp_path, [plain, ws])
    m = Manifest.load(out, pin=Manifest.load(out).digest)
    text = rust.emit(m, tmp_path / "launchers.rs", ops=[op]).read_text()
    assert "the block and the dynamic shared memory of the binary of each architecture" in text
    assert "- sm_103: block 256 threads (warp-specialised: 8 warps, 4 of them in its default partition), 16 bytes." in text
    assert "- sm_90: block 128 threads, 16 bytes." in text
    alone = export(tmp_path, ws, "alone")
    text = rust.emit(Manifest.load(alone, pin=Manifest.load(alone).digest), tmp_path / "alone.rs", ops=[op]).read_text()
    assert "block 256 threads (warp-specialised: 8 warps, 4 of them in its default partition), 16 bytes of dynamic shared memory." in text


def test_the_rust_emitter_refuses_what_the_strict_reader_refuses(ws_manifest, tmp_path):
    from tilefish.hosts import rust

    spec, out = ws_manifest
    m = Manifest.load(out, pin=Manifest.load(out).digest)
    entry = m.get(spec.stem, f"sm_{ARCH}").entry
    entry["total_num_warps"] = 4  # (edited after the reader checked it: what the emitter's own validation holds to)
    with pytest.raises(TilefishManifestError, match="total_num_warps must be an int above num_warps"):
        rust.emit(m, tmp_path / "launchers.rs")
