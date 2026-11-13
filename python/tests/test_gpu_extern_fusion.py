"""An extern's FMA contraction, on a real GPU: ``@tilefish.extern(enable_fp_fusion=...)``.

``gpu_kernels_p4.py`` declares one hand-written Triton kernel, ``y = a * b + c`` in f32, twice: ``fma_plain`` with ``enable_fp_fusion=False`` and
``fma_fused`` with ``enable_fp_fusion=True``. The option reaches Triton (the PTX has a multiply and an add, or one FMA), the bytes are those of two
roundings and of one, ``jit`` and ``replay`` of an exported manifest agree byte for byte, the manifest entry records the option, ``True`` is the
machine code of a declaration that says nothing (Triton's default), and a manifest built with the other setting is refused by ``replay``.
"""

import collections
import re

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import gpu_kernels_p4 as k  # noqa: E402
from gpu_util import differing, randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Out, Tensor, TilefishManifestError, const, f32, runtime  # noqa: E402
from tilefish.aot import build, compile_specs  # noqa: E402

CAP = torch.cuda.get_device_capability()
ARCH = CAP[0] * 10 + CAP[1]
F32 = torch.float32
N = (1 << 16) + 3  # not a multiple of the block: the mask is used


def instructions(ptx):
    """Instruction mnemonics of a PTX text and how often each occurs (predicates stripped)."""
    ops = collections.Counter()
    for line in ptx.splitlines():
        s = line.strip()
        if not s or s[0] in ".{}$()" or s.startswith("//"):
            continue
        m = re.match(r"(?:@!?%p\d+\s+)?([a-z][a-z0-9_.]*)", s)
        if m:
            ops[m.group(1)] += 1
    return ops


def built(op):
    (spec,) = compile_specs(op)
    return build(spec, arch=ARCH, fn=k.fma_kernel, cache=False)


def unset_declaration():
    """The same kernel declared with nothing about the contraction (Triton's default)."""

    @tilefish.extern(triton=k.fma_kernel, num_warps=4, grid=lambda L: (tilefish.cdiv(L, 1024),), name="fma_default")
    def fma_default(a: Tensor[k.L, f32], b: Tensor[k.L, f32], c: Tensor[k.L, f32], y: Out[Tensor[k.L, f32]], n: k.L, BLOCK: const = 1024): ...

    return fma_default


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("fusion-manifest")
    manifest = runtime.export([k.fma_plain, k.fma_fused], directory)
    return directory, manifest


def operands(seed):
    return randn((N,), F32, seed), randn((N,), F32, seed + 1), randn((N,), F32, seed + 2)


def test_the_option_reaches_triton():
    plain, fused, default = instructions(built(k.fma_plain).ptx), instructions(built(k.fma_fused).ptx), built(unset_declaration())
    assert plain["fma.rn.f32"] == 0 and plain["mul.rn.f32"] > 0 and plain["add.rn.f32"] > 0  # two instructions, two roundings
    assert fused["fma.rn.f32"] > 0 and fused["mul.rn.f32"] == 0 and fused["add.rn.f32"] == 0  # one instruction, one rounding
    assert built(k.fma_fused).sha256_code == default.sha256_code  # True is Triton's default: the same machine code as saying nothing
    assert built(k.fma_plain).sha256_code != default.sha256_code


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_bytes_are_two_roundings_or_one_and_jit_is_replay(exported, seed):
    directory, manifest = exported
    a, b, c = operands(seed)
    two_roundings = torch.mul(a, b).add(c)  # torch rounds the product, then the sum
    one_rounding = (a.double() * b.double() + c.double()).to(F32)  # the exact product and sum, rounded once
    results = {}
    for op in (k.fma_plain, k.fma_fused):
        jit = tilefish.run(op, a, b, c)
        replay = tilefish.run(op, a, b, c, mode="replay", manifest=directory, pin=manifest.digest)
        assert same_bytes(jit, replay), f"{op.name}: jit and replay differ"
        results[op.name] = replay
    assert same_bytes(results["fma_plain"], two_roundings)
    assert same_bytes(results["fma_fused"], one_rounding)
    differ = differing(results["fma_plain"], results["fma_fused"])
    print(f"seed {seed}: the contraction changes {differ} of {N} elements")
    assert differ > N // 20


def test_the_manifest_records_the_option_and_replay_refuses_the_other_setting(exported):
    directory, manifest = exported
    arch = f"sm_{ARCH}"
    assert manifest.get("fma_plain", arch).entry["options"] == {"num_warps": 4, "num_stages": 3, "enable_fp_fusion": False}
    assert manifest.get("fma_fused", arch).entry["options"] == {"num_warps": 4, "num_stages": 3, "enable_fp_fusion": True}

    # the kernel of fma_fused declared with the other setting under the same name: the manifest's binary is not this declaration's
    @tilefish.extern(triton=k.fma_kernel, num_warps=4, grid=lambda L: (tilefish.cdiv(L, 1024),), enable_fp_fusion=False, name="fma_fused")
    def other(a: Tensor[k.L, f32], b: Tensor[k.L, f32], c: Tensor[k.L, f32], y: Out[Tensor[k.L, f32]], n: k.L, BLOCK: const = 1024): ...

    a, b, c = operands(7)
    with pytest.raises(TilefishManifestError, match="is not what the compile spec asks for"):
        tilefish.run(other, a, b, c, mode="replay", manifest=directory, pin=manifest.digest)
