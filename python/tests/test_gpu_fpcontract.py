"""The contraction check on real binaries (:func:`tilefish._build.check_contraction`): a kernel built with ``enable_fp_fusion=False`` whose SASS
fused a multiply and an add is refused, on sm_90, sm_100, sm_103 and sm_120 (each built here, whatever the device: a build needs ptxas, not
the GPU; the SASS is read by the first ``nvdisasm`` that decodes the architecture, or the PTX is assembled for the base of its family).

* ``fpcontract_kernels.row_axpb_plain`` (Triton) and ``row_axpb_gluon`` (Gluon), ``y = x * w + b`` that LLVM packs into ``mul.rn.f32x2`` and
  ``add.rn.f32x2``, are refused on sm_100 and later with a message that names the kernel and the fixes, and build on sm_90; ``row_axpb_rn``
  (scalar ``.rn`` inline PTX) builds everywhere and rounds twice on the device; ``row_axpb_fused`` (fusion allowed) is never checked.
* Every lowered kernel of the case modules, with ``fp_fusion=False``, passes the check on every architecture: its float ``+ - *`` are scalar
  ``.rn`` instructions (note 2 of ``tilefish.lower``). So does every region of the fused graphs.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import fpcontract_kernels as fk  # noqa: E402
from conftest import ptxas_problem  # noqa: E402
from gpu_util import same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import _fpcontract, runtime  # noqa: E402
from tilefish.aot import build, compile_specs  # noqa: E402
from tilefish.decorators import KernelOp  # noqa: E402
from tilefish.errors import TilefishBuildError  # noqa: E402

CAP = torch.cuda.get_device_capability()
ARCH = CAP[0] * 10 + CAP[1]
ARCHS = [a for a in (90, 100, 103, 120) if ptxas_problem(a) is None]
PACKED = [a for a in ARCHS if a >= 100]  # LLVM packs f32 lanes from sm_100 on


def _build(op, arch, fn):
    (spec,) = compile_specs(op)
    return build(spec, arch=arch, fn=fn, cache=False)


@pytest.mark.parametrize("arch", ARCHS)
def test_a_contracted_triton_extern_is_refused_from_sm_100_on(arch):
    if arch not in PACKED:
        b = _build(fk.row_axpb_plain, arch, fk.row_axpb_kernel)  # no packed f32 instruction: nothing fused
        assert "f32x2" not in b.ptx
        return
    with pytest.raises(TilefishBuildError) as e:
        _build(fk.row_axpb_plain, arch, fk.row_axpb_kernel)
    msg = str(e.value)
    assert f"'row_axpb_plain' (fpcontract_kernels:row_axpb_kernel) for sm_{arch}" in msg
    assert "enable_fp_fusion=False" in msg and "lanes of fused multiply-adds" in msg and "where the same PTX with its multiplies kept apart has 0" in msg
    assert ("FFMA2" if arch in (100, 103) else "FFMA ") in msg
    assert '"mul.rn.f32 $0, $1, $2;"' in msg and "declare enable_fp_fusion=True" in msg


@pytest.mark.parametrize("arch", ARCHS)
def test_the_same_kernel_with_scalar_rn_instructions_builds_and_the_fused_declaration_is_not_checked(arch):
    b = _build(fk.row_axpb_rn, arch, fk.row_axpb_rn_kernel)
    assert "mul.rn.f32 " in b.ptx and "f32x2" not in b.ptx
    fused = _build(fk.row_axpb_fused, arch, fk.row_axpb_kernel)
    assert "fma.rn" in fused.ptx  # contracted by LLVM itself: allowed


@pytest.mark.parametrize("arch", ARCHS)
def test_a_contracted_gluon_extern_is_refused_from_sm_100_on(arch):
    if arch not in PACKED:
        _build(fk.row_axpb_gluon, arch, fk.row_axpb_gluon_kernel)
        return
    with pytest.raises(TilefishBuildError, match=r"'row_axpb_gluon' \(fpcontract_kernels:row_axpb_gluon_kernel\) for sm_\d+ .* fused multiplies and adds"):
        _build(fk.row_axpb_gluon, arch, fk.row_axpb_gluon_kernel)


def test_the_kernel_that_keeps_its_roundings_rounds_twice_on_the_device():
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(4096, 1024, generator=g, device="cuda").to(torch.bfloat16)
    w = torch.randn(1024, generator=g, device="cuda").to(torch.bfloat16)
    b = torch.randn(1024, generator=g, device="cuda").to(torch.bfloat16)
    got = tilefish.run(fk.row_axpb_rn, x, w, b)
    assert same_bytes(got, ((x.float() * w.float()) + b.float()).to(torch.bfloat16))  # torch rounds the product and the sum each in f32


def _case_kernels():
    import broadcast_cases
    import cast_cases
    import ieee_cases
    import lower_cases
    import mask_cases
    import primitive_cases
    import slice_cases
    import tanh_cases

    out = {}
    for mod in (lower_cases, ieee_cases, primitive_cases, broadcast_cases, cast_cases, mask_cases, slice_cases, tanh_cases):
        for name, k in vars(mod).items():
            if isinstance(k, KernelOp):
                out[f"{mod.__name__}.{name}"] = k
    return out


@pytest.mark.parametrize("arch", ARCHS)
def test_every_lowered_case_with_fp_fusion_false_passes_the_check_on_every_architecture(arch):
    refused, built = [], 0
    for name, k in sorted(_case_kernels().items()):
        try:
            low = tilefish.lower(k, fp_fusion=False)
        except Exception:  # noqa: BLE001 - a case that only lowers with its own options (or not at all: a refusal case)
            continue
        fn = runtime.load_lowered(low)
        for spec in compile_specs(low.extern)[:2]:
            try:
                build(spec, arch=arch, fn=fn)
                built += 1
            except TilefishBuildError as e:
                if "fused multiplies and adds" in str(e):
                    refused.append(f"{name}: {e}")
                elif "cannot" not in str(e):  # a case for one architecture only (an sm_100 instruction) is not this test's
                    raise
    assert built > 40 and not refused, refused


def test_the_check_reads_the_binary_of_the_device_or_its_family():
    b = _build(fk.row_axpb_rn, ARCH, fk.row_axpb_rn_kernel)
    v = _fpcontract.check(b.ptx, b.cubin, b.options, key=b.sha256)
    assert v is not None and not v.contracted and v.lanes == v.allowed == 0
    assert v.target in (b.ptx_target, {"sm_103a": "sm_100a", "sm_121a": "sm_120a"}.get(b.ptx_target))
    assert v.proxy == (v.target != b.ptx_target)
