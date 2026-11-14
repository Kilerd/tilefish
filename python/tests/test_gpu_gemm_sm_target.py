"""A GEMM pinned for a partition of the device's SMs, on a real GPU: ``sm_count_target`` and green contexts.

* A target of the device's own count pins what no target pins.
* The bytes of a pinned algorithm do not depend on the target it runs with.
* A declaration with ``sm_count_target=16``: ``jit`` and ``replay`` agree byte for byte, the entry records it.
* In a process whose first cuBLASLt handle was made in a **green context** of 16 SMs (an SM partition made with the driver's green-context
  API), the pins with ``sm_count_target=16`` are the pins of a process whose first handle was made in the primary context, byte for byte; the
  pins without a target are counted (they may differ: the heuristic's answers depend on the context of the process's first handle, measured
  with cuBLASLt 12.8.4, ``docs/manifest-schema-3.md``, section 9.3), and printed.
"""

import json
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, f16, f32, runtime  # noqa: E402
from tilefish import _cublaslt as L  # noqa: E402

TARGET = 16
SHAPES = [(1, 1024, 4096), (16, 512, 8192), (64, 3072, 3072), (128, 1024, 4096), (512, 1536, 3072), (2048, 1024, 1024), (8192, 512, 3072)]
FORMS = [("TN", False), ("NN", False), ("NN", True)]


def sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


@pytest.fixture(autouse=True, scope="module")
def _cuda_is_up():
    torch.zeros(1, device="cuda")
    if sms() <= TARGET:
        pytest.skip(f"the device has {sms()} SMs: too few for a partition of {TARGET}")


def pins(target):
    return {
        f"{layout}:{bias}:{m}x{n}x{k}": L.pin_algo(m, n, k, bias, form=L.Form("f16", "f16", layout), workspace_bytes=64 << 20, sm_count_target=target).algo_hex
        for layout, bias in FORMS
        for m, n, k in SHAPES
    }


def test_a_target_of_the_devices_count_pins_what_no_target_pins():
    assert pins(sms()) == pins(None)


def test_the_bytes_of_a_pinned_algorithm_do_not_depend_on_the_target_it_runs_with():
    for layout, bias in FORMS:
        for m, n, k in SHAPES[:4]:
            form = L.Form("f16", "f16", layout)
            pin = L.pin_algo(m, n, k, bias, form=form, workspace_bytes=64 << 20, sm_count_target=TARGET)
            a = randn((m, k), torch.float16, seed=m)
            b = randn((n, k) if layout == "TN" else (k, n), torch.float16, seed=m + 1, scale=1 / k**0.5)
            c = randn((n,), torch.float16, seed=m + 2) if bias else None
            outs = []
            for target in (TARGET, None, sms()):
                d = torch.empty((m, n), dtype=torch.float16, device="cuda")
                L.matmul(
                    a.data_ptr(), b.data_ptr(), d.data_ptr(), None if c is None else c.data_ptr(), m, n, k, pin.algo_hex,
                    workspace_bytes=64 << 20, stream=torch.cuda.current_stream().cuda_stream, form=form, sm_count_target=target,
                )  # fmt: skip
                outs.append(d)
            torch.cuda.synchronize()
            assert same_bytes(outs[0], outs[1]) and same_bytes(outs[0], outs[2]), (layout, bias, m, n, k)


def test_a_declaration_with_a_target_runs_the_same_in_jit_and_replay(tmp_path):
    M = Dim("M", buckets=[16, 512])

    @tilefish.library("part", acc=f32, impl="cublaslt.gemm_bias", layout="NN", workspace_bytes=64 << 20, sm_count_target=TARGET)
    def part(a: Tensor[M, 3072, f16], b: Tensor[3072, 1536, f16], bias: Tensor[1536, f16]) -> Tensor[M, 1536, f16]: ...

    manifest = runtime.export([part], tmp_path)
    for e in manifest.library_entries:
        assert e["sm_count_target"] == TARGET and e["sm_count"] == sms()
    for m in (16, 512):
        args = (randn((m, 3072), torch.float16, seed=m), randn((3072, 1536), torch.float16, seed=1, scale=0.02), randn((1536,), torch.float16, seed=2))
        jit = tilefish.run(part, *args)
        assert same_bytes(jit, tilefish.run(part, *args, mode="replay", manifest=tmp_path, pin=manifest.digest)), m


GREEN = textwrap.dedent(
    """
    import ctypes, json, sys
    import torch
    from cuda.bindings import driver as drv
    from tilefish import _cublaslt as L
    sys.path.insert(0, {tests!r})
    import test_gpu_gemm_sm_target as T

    def ok(r):
        assert r[0] == drv.CUresult.CUDA_SUCCESS, r[0]
        return r[1] if len(r) == 2 else r[1:]

    torch.zeros(1, device="cuda")  # torch's primary context; no cuBLASLt handle exists yet (no matmul ran)
    if not L._mapped_paths() and L._wheel_path():  # (open torch's copy without a matmul, which would make a handle of its own first)
        L._library = L._Library(L._wheel_path())
    if not L._mapped_paths():
        print(json.dumps({{"skip": "no libcublasLt can be opened before the first handle"}}))
        sys.exit(0)
    lib = L._lib()
    SM = drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
    dev = ok(drv.cuDeviceGet(0))
    groups, count, _ = ok(drv.cuDevSmResourceSplitByCount(1, ok(drv.cuDeviceGetDevResource(dev, SM)), 0, {target}))
    green = ok(drv.cuGreenCtxCreate(ok(drv.cuDevResourceGenerateDesc([groups[0]], 1)), dev, drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM))
    got = int(ok(drv.cuGreenCtxGetDevResource(green, SM)).sm.smCount)
    primary = ok(drv.cuCtxGetCurrent())
    ok(drv.cuCtxSetCurrent(ok(drv.cuCtxFromGreenCtx(green))))
    handle = ctypes.c_void_p()
    lib.call("cublasLtCreate", ctypes.byref(handle))  # the process's first cuBLASLt handle, made in the green context
    ok(drv.cuCtxSetCurrent(primary))
    print(json.dumps({{"green_sms": got, "target": T.pins({target}), "none": T.pins(None)}}))
    """
)


def test_in_a_process_whose_first_handle_is_in_a_green_context_a_target_pins_what_it_pins_elsewhere(capsys):
    import os

    tests = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run([sys.executable, "-c", GREEN.format(tests=tests, target=TARGET)], capture_output=True, text=True, check=True)
    got = json.loads(out.stdout.strip().splitlines()[-1])
    if "skip" in got:
        pytest.skip(got["skip"])
    assert got["green_sms"] == TARGET
    here_target, here_none = pins(TARGET), pins(None)
    assert got["target"] == here_target  # the target makes the pin the same wherever the first handle was made
    differ = sum(got["none"][k] != here_none[k] for k in here_none)
    with capsys.disabled():
        print(
            f"\n[green context] first handle in a partition of {TARGET} of {sms()} SMs: with sm_count_target={TARGET}, {len(here_target)} of "
            f"{len(here_target)} pins equal the primary context's; without a target, {differ} of {len(here_none)} differ"
        )
