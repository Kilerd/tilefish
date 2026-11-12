"""What the CUDA driver takes, on a real GPU: shared memory, the device a kernel is loaded for, what a refused kernel leaves behind.

* The shared memory a launch asks for is checked against what the driver enforces: the function's *static* shared memory
  (``CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES``, as the driver reports it once the module is loaded) plus the dynamic size must not
  exceed the device's opt-in maximum per block. Measured on an sm_103 device: a Triton kernel's ELF has a 1024-byte
  ``.nv.shared.<symbol>`` section, but the driver reports 0 static bytes and takes a dynamic size of exactly the opt-in maximum; a
  kernel with 4096 bytes of static shared memory takes the opt-in maximum minus 4096 and not a byte more.
* A dynamic size above 48 KiB needs ``cuFuncSetAttribute``; a ``tl.dot`` kernel of about 128 KiB runs in both modes.
* A kernel is loaded into the context of the device it is for, and one the driver refuses is not left loaded.
"""

import dataclasses
import hashlib
import importlib
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
from tilefish import Dim, Out, Tensor, TilefishRuntimeError, _cuda, f32, runtime  # noqa: E402
from tilefish.aot import build  # noqa: E402
from tilefish.dims import as_dim, cdiv  # noqa: E402
from tilefish.signature import Param, Signature  # noqa: E402

CAP = torch.cuda.get_device_capability()
ARCH = f"sm_{CAP[0] * 10 + CAP[1]}"


def extern_with(element_counts=None, params=None, grid=None, name="probe"):
    """An ``ExternOp`` made the way the lowering makes one (no check that the dimensions are derivable)."""
    from tilefish.decorators import ExternOp

    def stub():
        pass

    stub.__name__ = name
    return ExternOp(stub, name, Signature(params, ()), "no_such_module:no_such_kernel", 4, 3, grid, element_counts=element_counts)


# -- the shared memory the driver takes ------------------------------------------------------------------------------------------------------------------


def device_optin():
    drv = _cuda._driver()
    torch.zeros(1, device="cuda")
    (device,) = _cuda._check(drv.cuCtxGetDevice(), "cuCtxGetDevice")
    (optin,) = _cuda._check(
        drv.cuDeviceGetAttribute(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device), "optin"
    )
    return int(optin)


def softmax_prepared():
    x = randn((7, 1000), torch.float32, 0)
    target = runtime._target(softmax)
    prep = runtime._prepare(target, (x,), {})
    low = target.lowered
    built = build(prep.spec, arch=ARCH, fn=runtime.load_lowered(low))
    return x, prep, built, tilefish.run(softmax, x, mode="jit")


def test_the_driver_takes_a_dynamic_size_of_exactly_the_optin_maximum_for_a_triton_kernel():
    """Its ELF section ``.nv.shared.<symbol>`` (1024 bytes on sm_90 and up) is not static shared memory: the driver reports 0."""
    limit = device_optin()
    x, prep, built, expected = softmax_prepared()
    assert limit > _cuda.DEFAULT_SHARED_LIMIT  # so the cuFuncSetAttribute path is the one that runs
    runtime._launch_built(prep, dataclasses.replace(built, shared=limit), 0)
    torch.cuda.synchronize()
    assert same_bytes(prep.results[0], expected)
    loaded = _cuda.load(dataclasses.replace(built, shared=limit), 0)
    assert loaded.static_shared == 0
    with pytest.raises(TilefishRuntimeError, match=rf"needs 0 bytes of static and {limit + 1} bytes of dynamic shared memory, more than the {limit} bytes a block may use"):
        _cuda.load(dataclasses.replace(built, shared=limit + 1), 0)
    torch.cuda.synchronize()  # the driver was not asked: the context is fine
    assert torch.isfinite(tilefish.run(softmax, x, mode="jit")).all()


def test_a_kernel_with_static_shared_memory_leaves_the_dynamic_size_its_share(tmp_path):
    """Measured: static 4096 -> the driver takes at most optin - 4096 dynamic, and not a byte more."""
    from tilefish._build import LaunchAttributes, BuiltKernel

    limit = device_optin()
    static = 4096
    _, _, softmax_built, _ = softmax_prepared()
    version = [ln for ln in softmax_built.ptx.splitlines() if ln.startswith(".version")][0].split()[1]
    ptx = (
        f".version {version}\n.target {softmax_built.ptx_target}\n.address_size 64\n"
        ".visible .entry kern(.param .u64 p, .param .u64 s0, .param .u64 s1)\n{\n"
        f"    .reg .b32 r;\n    .reg .b64 a;\n    .shared .align 16 .b8 buf[{static}];\n"
        "    ld.param.u64 a, [p];\n    mov.u32 r, 7;\n    st.shared.u32 [buf], r;\n    ld.shared.u32 r, [buf];\n"
        "    st.global.u32 [a], r;\n    ret;\n}\n"
    )
    (tmp_path / "k.ptx").write_text(ptx)
    ptxas = str(triton.knobs.nvidia.ptxas.path)
    subprocess.run([ptxas, f"-arch={softmax_built.ptx_target}", str(tmp_path / "k.ptx"), "-o", str(tmp_path / "k.cubin")], check=True, capture_output=True)
    cubin = (tmp_path / "k.cubin").read_bytes()

    M = Dim("M")
    ext = extern_with(None, [Param("out", 0, "tensor", "out", type=Tensor[M, tilefish.i32])], (as_dim(M),), name="static_probe")
    ext_spec = tilefish.aot.compile_specs(ext)[0]

    def built(dynamic):
        return BuiltKernel(
            spec=ext_spec, arch=ARCH, cubin=cubin, ptx="", symbol="kern", shared=dynamic, num_warps=ext_spec.num_warps,
            sha256=hashlib.sha256(cubin).hexdigest(), sha256_code="0" * 64, triton_version="", ptxas_version="", ptx_target="",
            options={}, launch=LaunchAttributes(0, 0, False, (1, 1, 1), 1, False),
        )

    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    fits = limit - static
    for dynamic in (0, 1024, fits - 1, fits):
        loaded = _cuda.load(built(dynamic), 0)
        assert loaded.static_shared == static
        out.zero_()
        loaded.launch((1,), [("pointer", out.data_ptr())], stream)
        torch.cuda.synchronize()
        assert int(out[0]) == 7, dynamic
    with pytest.raises(TilefishRuntimeError, match=rf"needs {static} bytes of static and {fits + 1} bytes of dynamic shared memory, more than the {limit} bytes"):
        _cuda.load(built(fits + 1), 0)
    with pytest.raises(TilefishRuntimeError, match=rf"needs {static} bytes of static and {limit} bytes of dynamic"):
        _cuda.load(built(limit), 0)  # what a triton kernel may ask for is too much once there is static memory
    torch.cuda.synchronize()


def test_a_kernel_the_driver_refuses_is_not_left_loaded(monkeypatch):
    drv = _cuda._driver()
    _, prep, built, _ = softmax_prepared()
    limit = device_optin()
    loads, unloads = [], []
    real_load, real_unload = drv.cuModuleLoadData, drv.cuModuleUnload
    monkeypatch.setattr(drv, "cuModuleLoadData", lambda blob: loads.append(1) or real_load(blob))
    monkeypatch.setattr(drv, "cuModuleUnload", lambda module: unloads.append(1) or real_unload(module))
    before = dict(_cuda._loaded)
    with pytest.raises(TilefishRuntimeError, match="more than the"):
        _cuda.load(dataclasses.replace(built, shared=limit + 1), 0)  # refused after the module was loaded
    assert (len(loads), len(unloads)) == (1, 1)
    other = runtime._target(gelu_exact).compile_specs()[0]  # a spec whose ABI is not the binary's: _check_params refuses
    with pytest.raises(TilefishRuntimeError, match=r"takes parameters of \[.*\] bytes, but the ABI in the manifest says"):
        _cuda.load(dataclasses.replace(built, spec=other), 0)
    assert (len(loads), len(unloads)) == (2, 2) and _cuda._loaded == before  # nothing is cached for either


def test_a_kernel_is_loaded_into_the_context_of_its_device():
    _, _, built, _ = softmax_prepared()
    before = dict(_cuda._loaded)
    with pytest.raises(TilefishRuntimeError, match=r"is to be loaded for device 1, but the current CUDA context is on device 0: make that device current"):
        _cuda.load(built, 1)
    assert _cuda._loaded == before


# -- more than 48 KiB of dynamic shared memory: a hand-written tl.dot kernel ---------------------------------------------------------------------------------

MM_SOURCE = """
import triton
import triton.language as tl


@triton.jit
def mm_kernel(a_ptr, b_ptr, c_ptr, m, n, k, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, k, BK):
        a = tl.load(a_ptr + offs_m[:, None] * k + (kk + offs_k)[None, :])
        b = tl.load(b_ptr + (kk + offs_k)[:, None] * n + offs_n[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], acc)
"""

MmM, MmN, MmK = Dim("MmM", divisible_by=128), Dim("MmN", divisible_by=128), Dim("MmK", divisible_by=64)


@tilefish.extern(triton="mm_mod:mm_kernel", num_warps=8, num_stages=4, grid=lambda MmM, MmN: (cdiv(MmM, 128), cdiv(MmN, 128)))
def mm(
    a_ptr: Tensor[MmM, MmK, tilefish.f16],
    b_ptr: Tensor[MmK, MmN, tilefish.f16],
    c_ptr: Out[Tensor[MmM, MmN, f32]],
    m: MmM,
    n: MmN,
    k: MmK,
    *,
    BM: tilefish.const = 128,
    BN: tilefish.const = 128,
    BK: tilefish.const = 64,
): ...


@pytest.fixture
def mm_dir(tmp_path, monkeypatch):
    d = tmp_path / "mm"
    d.mkdir()
    (d / "mm_mod.py").write_text(MM_SOURCE)
    monkeypatch.syspath_prepend(str(d))
    sys.modules.pop("mm_mod", None)
    importlib.invalidate_caches()
    yield d
    sys.modules.pop("mm_mod", None)


def test_a_kernel_with_more_than_48_kib_of_dynamic_shared_memory_runs_in_jit_and_replay(mm_dir, tmp_path):
    """``cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)`` is what lets the launch ask for more than the default 48 KiB."""
    torch.manual_seed(0)
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    manifest = runtime.export(mm, tmp_path / "m")
    entry = manifest.get("mm", ARCH).entry
    shared = entry["shared"]
    assert shared > _cuda.DEFAULT_SHARED_LIMIT, f"the kernel needs more than 48 KiB of shared memory for this to test anything: {shared}"
    assert entry["launch"]["global_scratch_size"] == 0 and entry["launch"]["cluster_dims"] == [1, 1, 1]  # a plain cuLaunchKernel can run it
    reference = a.float() @ b.float()
    jit = tilefish.run(mm, a, b, mode="jit")
    replay = tilefish.run(mm, a, b, mode="replay", manifest=manifest.directory, pin=manifest.digest)
    assert torch.allclose(jit, reference, rtol=1e-3, atol=2e-2) and same_bytes(jit, replay)
    # the function was given the room before it was launched: the driver's own attribute says so
    (spec,) = [s for s in runtime._target(mm).compile_specs()]
    built = manifest.built_kernel(spec, ARCH, pinned=True)  # (the source of the kernel is not what this test is about)
    loaded = _cuda.load(built, 0)
    drv = _cuda._driver()
    (granted,) = _cuda._check(drv.cuFuncGetAttribute(drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, loaded.function), "attr")
    assert int(granted) == shared > _cuda.DEFAULT_SHARED_LIMIT


def test_a_size_that_needs_no_opt_in_does_not_ask_for_one():
    """At most 48 KiB is the default: the function's limit is left alone, and the launch with exactly that size works."""
    x, prep, built, expected = softmax_prepared()
    drv = _cuda._driver()
    for shared in (16, _cuda.DEFAULT_SHARED_LIMIT):
        loaded = _cuda.load(dataclasses.replace(built, shared=shared), 0)
        (granted,) = _cuda._check(drv.cuFuncGetAttribute(drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, loaded.function), "attr")
        assert int(granted) == _cuda.DEFAULT_SHARED_LIMIT
        runtime._launch_built(prep, dataclasses.replace(built, shared=shared), 0)
    torch.cuda.synchronize()
    assert same_bytes(prep.results[0], expected)
    above = _cuda.load(dataclasses.replace(built, shared=_cuda.DEFAULT_SHARED_LIMIT + 1), 0)
    (granted,) = _cuda._check(drv.cuFuncGetAttribute(drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, above.function), "attr")
    assert int(granted) == _cuda.DEFAULT_SHARED_LIMIT + 1
