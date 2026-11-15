"""The TF32 guard on a real GPU and a real Triton (the kernels of ``tf32_kernels.py``; ``test_tf32guard.py`` has the detector on IR text).

An f32 ``tl.dot(..., input_precision="ieee")`` whose operand ``a`` was loaded as f16 is compiled by Triton 3.5.1 into ``tcgen05.mma ... kind::tf32``
on sm_100 and sm_103 when its tile has 64 rows, and onto FMAs at 32 rows or on sm_90 and sm_120. The operands are made so that every product and
sum is exact in f32 (``a = 1``, ``b = 1 + 2**-12``, 16 terms): the IEEE dot gives ``16 + 2**-8`` in every output and a TF32 MMA ``16`` (TF32 keeps
10 bits of ``b``'s mantissa, so ``b`` is 1). Here:

* the trap is real on this device (sm_100 or sm_103): Triton's own launch of the 64-row IEEE kernel gives 16, and on random data almost every
  output differs from the 32-row kernel's, which is within an f32 rounding of a float64 product;
* tilefish refuses to build it for sm_100 and sm_103 (a compile only, any device), and ``jit`` and ``export`` refuse it on this device; sm_90
  and sm_120 keep it on FMAs and build it;
* the same kernel at 32 rows, and at 64 rows with ``input_precision="tf32"``, build and run (the second gives TF32's 16, which it asked for);
* a kernel with an IEEE dot and a TF32 dot is judged dot by dot: refused when the IEEE dot went to tcgen05, built when only the TF32 dot did.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import tf32_kernels as tk  # noqa: E402
from conftest import require_ptxas_for  # noqa: E402
from gpu_util import differing, randn  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishBuildError, _tf32guard, runtime  # noqa: E402
from tilefish.aot import build, compile_specs  # noqa: E402

CAP = torch.cuda.get_device_capability()
CC = CAP[0] * 10 + CAP[1]
TCGEN05 = CC in (100, 103)  # the architectures whose tensor cores Triton 3.5.1 sends the IEEE dot to
ROWS = 256
EXACT = 16 * (1 + 2**-12)  # 16 + 2**-8, exact in f32
TF32 = 16.0
REFUSED = (
    r"asks for 1 f32 dot with IEEE input precision, and was compiled into TF32 tensor-core MMAs: \d+ x tcgen05\.mma\.cta_group::1\.kind::tf32 in "
    r"its PTX; 1 x ttng\.tc_gen5_mma of f32 operands in its TTGIR \(the dot at \S*tf32_kernels\.py:\d+:\d+\)"
)
on_tcgen05 = pytest.mark.skipif(not TCGEN05, reason=f"the trap is Triton 3.5.1's on sm_100 and sm_103; this device is sm_{CC}")


def exact_operands(rows=ROWS):
    a = torch.ones((rows, 16), device="cuda", dtype=torch.float16)
    b = torch.full((16, 32), 1 + 2**-12, device="cuda", dtype=torch.float32)
    return a, b


def triton_launch(a, b, bm, prec):
    """The kernel launched by Triton itself, with no tilefish in the way."""
    c = torch.empty((a.shape[0], 32), device="cuda", dtype=torch.float32)
    tk.dot_kernel[(a.shape[0] // bm,)](a, b, c, BM=bm, PREC=prec, num_warps=4)
    torch.cuda.synchronize()
    return c


@on_tcgen05
def test_the_trap_is_real_on_this_device():
    a, b = exact_operands()
    assert bool((triton_launch(a, b, 64, 0) == TF32).all())  # asked for IEEE, computed in TF32
    assert bool((triton_launch(a, b, 32, 0) == EXACT).all())  # 32 rows: FMAs, exact
    a, b = randn((ROWS, 16), torch.float16, seed=1), randn((16, 32), torch.float32, seed=2)
    want = a.double() @ b.double()
    tc, fma = triton_launch(a, b, 64, 0), triton_launch(a, b, 32, 0)
    assert differing(tc, fma) > 0.9 * tc.numel()  # the bits silently change
    assert torch.allclose(fma.double(), want, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(tc.double(), want, rtol=1e-5, atol=1e-5)  # TF32: about 3 decimal digits of b


@pytest.mark.parametrize("arch", [100, 103])
def test_tilefish_refuses_to_build_the_ieee_dot_of_64_rows_for_sm_100_and_sm_103(arch):
    require_ptxas_for(arch)
    (spec,) = compile_specs(tk.ieee64, arch=arch)
    with pytest.raises(TilefishBuildError, match=REFUSED) as raised:
        build(spec, arch=arch, cache=False)
    message = str(raised.value)
    assert message.startswith(f"'tf32_guard_ieee64' (tf32_kernels:dot_kernel) for sm_{arch} with Triton 3.5.1: it asks for")
    assert "fewer than 64 rows" in message and "tl.fma" in message and 'input_precision="tf32"' in message


@pytest.mark.parametrize("arch", [90, 120])
def test_sm_90_and_sm_120_keep_it_on_fmas(arch):
    require_ptxas_for(arch)
    (spec,) = compile_specs(tk.ieee64, arch=arch)
    built = build(spec, arch=arch, cache=False)
    assert _tf32guard.ptx_tf32_mmas(built.ptx) == [] and "fma.rn.f32" in built.ptx


@on_tcgen05
def test_jit_and_export_refuse_it_on_this_device(tmp_path):
    a, b = exact_operands()
    with pytest.raises(TilefishBuildError, match=REFUSED):
        tilefish.run(tk.ieee64, a, b)
    with pytest.raises(TilefishBuildError, match=REFUSED):
        runtime.export(tk.ieee64, tmp_path / "out")
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_32_rows_build_and_are_exact(tmp_path):
    require_ptxas_for(100, 103, CC)
    for arch in (100, 103):
        (spec,) = compile_specs(tk.ieee32, arch=arch)
        assert _tf32guard.ptx_tf32_mmas(build(spec, arch=arch, cache=False).ptx) == []
    a, b = exact_operands()
    jit = tilefish.run(tk.ieee32, a, b)
    assert bool((jit == EXACT).all())
    manifest = runtime.export(tk.ieee32, tmp_path / "out")
    replay = tilefish.run(tk.ieee32, a, b, mode="replay", manifest=tmp_path / "out", pin=manifest.digest)
    assert torch.equal(replay, jit)


def test_tf32_declared_builds_and_computes_tf32(tmp_path):
    require_ptxas_for(100, 103, CC)
    for arch in (100, 103):
        (spec,) = compile_specs(tk.tf32_64, arch=arch)
        assert "tcgen05.mma.cta_group::1.kind::tf32" in _tf32guard.ptx_tf32_mmas(build(spec, arch=arch, cache=False).ptx)
    a, b = exact_operands()
    jit = tilefish.run(tk.tf32_64, a, b)
    if TCGEN05:
        assert bool((jit == TF32).all())  # what it asked for
    manifest = runtime.export(tk.tf32_64, tmp_path / "out")
    assert torch.equal(tilefish.run(tk.tf32_64, a, b, mode="replay", manifest=tmp_path / "out", pin=manifest.digest), jit)


def test_a_kernel_with_an_ieee_and_a_tf32_dot_is_judged_dot_by_dot():
    require_ptxas_for(100, 103, CC)
    for arch in (100, 103):
        (spec,) = compile_specs(tk.mixed_f16, arch=arch)
        with pytest.raises(TilefishBuildError, match=r"1 TF32 tensor-core MMA of its TTGIR is made of it: 1 x ttng\.tc_gen5_mma of f32 operands"):
            build(spec, arch=arch, cache=False)
        (spec,) = compile_specs(tk.mixed_f32, arch=arch)
        built = build(spec, arch=arch, cache=False)
        assert "tcgen05.mma.cta_group::1.kind::tf32" in _tf32guard.ptx_tf32_mmas(built.ptx)  # the TF32 dot's
    a = torch.full((ROWS, 64), 1 + 2**-12, device="cuda", dtype=torch.float32)
    h = torch.ones((ROWS, 16), device="cuda", dtype=torch.float16)
    b = torch.ones((64, 32), device="cuda", dtype=torch.float32)
    c, d = tilefish.run(tk.mixed_f32, a, h, b)
    assert bool((c == EXACT).all())  # the IEEE dot, on FMAs: 16 + 2**-8
    if TCGEN05:
        assert bool((d == 64.0).all())  # the TF32 dot: a rounded to 1, 64 terms
