"""The roundings the contract writes out (notes 1, 2, 16 and 17 of ``tilefish.lower``), on a real GPU, against what each promises on the bits.

* ``-x`` gives the canonical NaN (``0x7FFFFFFF``, ``0x7FFF``) for every NaN and flips the sign of everything else: alone, as an operand of a
  ``where`` either side (which ptxas would fold into its select as a negated operand, flipping a NaN's sign), negated twice, in a default kernel,
  in a kernel with ``fp_fusion=False`` and in a graph's fused region; on every NaN of f32 and every bf16 and f16 pattern.
* A narrowing cast of a widened value (``x.to(f32).to(bf16)``, and through a ``where``) gives the canonical NaN for every NaN in a default kernel,
  as the cast alone does (no fold keeps the input's payload), and the input's bits for every other value.
* ``fp_fusion=False``: ``x * w + b`` of f32 rows rounds twice (torch's ``x * w + b``, two kernels), where LLVM packs the lanes and ptxas would
  contract them on sm_100 and later; ``fp_fusion=True`` may round once.

Nothing here is specific to sm_103 (an sm_120 GPU runs the same tests).
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

from gpu_util import differing, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, bf16, f16, f32, u8  # noqa: E402
from tilefish import _fpcontract, runtime  # noqa: E402
from tilefish.aot import compile_specs  # noqa: E402

CUDA = "cuda"
N = Dim("N")
R = Dim("R")
G32 = Dim("G32", buckets=[2 * ((1 << 23) - 1) + (1 << 22)])  # every f32 NaN and 2**22 random patterns: one bucket, the size of the input
G16 = Dim("G16", buckets=[16 << 16])  # every 16-bit pattern 16 times
TORCH = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
BITS = {"f32": torch.int32, "bf16": torch.int16, "f16": torch.int16}
CANON = {"f32": 0x7FFFFFFF, "bf16": 0x7FFF, "f16": 0x7FFF}
TYPES = {"f32": f32, "bf16": bf16, "f16": f16}


def every16(dtype):
    return torch.arange(-(2**15), 2**15, dtype=torch.int32, device=CUDA).to(torch.int16).view(dtype)


def f32_inputs():
    """Every f32 NaN (both signs, every payload), and 2**22 random bit patterns."""
    payload = torch.arange(1, 1 << 23, dtype=torch.int64, device=CUDA)
    nans = torch.cat([0x7F800000 + payload, -(2**31) + 0x7F800000 + payload]).to(torch.int32)
    g = torch.Generator(device=CUDA).manual_seed(11)
    rnd = torch.randint(-(2**31), 2**31, (1 << 22,), generator=g, device=CUDA, dtype=torch.int64).to(torch.int32)
    return torch.cat([nans, rnd]).view(torch.float32)


def negated(x, dt):
    """CUDA's ``-x`` on the bits: the sign flipped, the canonical NaN for a NaN."""
    b = x.view(BITS[dt])
    flipped = b ^ torch.tensor(-(2**31) if dt == "f32" else -(2**15), dtype=BITS[dt], device=CUDA)
    canon = torch.tensor(CANON[dt], dtype=BITS[dt], device=CUDA)
    return torch.where(torch.isnan(x.float()), canon, flipped).view(x.dtype)


def masks(n):
    g = torch.Generator(device=CUDA).manual_seed(5)
    return torch.randint(0, 2, (n,), generator=g, device=CUDA).to(torch.uint8)


def _neg_kernels(dt, fusion):
    T = TYPES[dt]

    @tilefish.kernel(fp_fusion=fusion, name=f"negsel_{dt}_{fusion}")
    def negsel(a: Tensor[N, T], b: Tensor[N, T], m: Tensor[N, u8]):
        keep = m != 0
        return -a, tilefish.where(keep, -a, b), tilefish.where(keep, b, -a), -(-a)

    return negsel


def _neg_graph(dt):
    T = TYPES[dt]
    G = G32 if dt == "f32" else G16

    def body(a, b, m):
        keep = m != 0
        return -a, tilefish.where(keep, -a, b), tilefish.where(keep, b, -a), -(-a)

    import inspect

    def fn(**kw):
        return body(kw["a"], kw["b"], kw["m"])

    fn.__signature__ = inspect.Signature(
        [inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=t) for p, t in (("a", Tensor[G, T]), ("b", Tensor[G, T]), ("m", Tensor[G, u8]))]
    )
    fn.__name__ = f"gnegsel_{dt}"
    return tilefish.graph(fn, name=f"gnegsel_{dt}")


def _check_neg(got, a, b, m, dt):
    keep = m != 0
    na = negated(a, dt)
    want = [na, torch.where(keep, na, b), torch.where(keep, b, na), negated(na, dt)]
    for k, (x, y) in enumerate(zip(got, want)):
        assert same_bytes(x, y), f"{dt} output {k}: {differing(x, y)} differ"


@pytest.mark.parametrize("dt", ["f32", "bf16", "f16"])
@pytest.mark.parametrize("where", ["default", "no-fusion", "region"])
def test_a_negation_gives_the_canonical_nan_alone_and_in_a_select(dt, where):
    a = f32_inputs() if dt == "f32" else every16(TORCH[dt]).repeat(16)
    b = a.flip(0).clone()
    m = masks(a.numel())
    if where == "region":
        got = tilefish.run(_neg_graph(dt), a, b, m)
    else:
        got = tilefish.run(_neg_kernels(dt, where == "default"), a, b, m)
    _check_neg(got, a, b, m, dt)


@pytest.mark.parametrize("src,narrow", [("bf16", "bf16"), ("f16", "f16"), ("f16", "bf16"), ("bf16", "f16")])
def test_a_narrowing_cast_of_a_widened_value_gives_the_canonical_nan_in_a_default_kernel(src, narrow):
    S, T = TYPES[src], TYPES[narrow]

    @tilefish.kernel(name=f"roundtrip_{src}_{narrow}")
    def roundtrip(x: Tensor[N, S], y: Tensor[N, f32], m: Tensor[N, u8]):
        wx = x.to(f32)
        return wx.to(T), tilefish.where(m != 0, wx, y).to(T)

    x = every16(TORCH[src]).repeat(4)
    y = torch.randn(x.numel(), device=CUDA)
    m = masks(x.numel())
    got = tilefish.run(roundtrip, x, y, m)
    one = x.float().to(TORCH[narrow])  # torch's CUDA cast (cvt.rn): the canonical NaN for every NaN
    canon = torch.tensor(0x7FFF, dtype=torch.int16, device=CUDA).view(TORCH[narrow])
    one = torch.where(torch.isnan(one.float()), canon, one)
    sel = torch.where(m != 0, one, torch.where(torch.isnan(y), canon, y.to(TORCH[narrow])))
    assert same_bytes(got[0], one), differing(got[0], one)
    assert same_bytes(got[1], sel), differing(got[1], sel)
    low = tilefish.lower(roundtrip)
    assert low.source.count(f" = _tf_cvt_{narrow}(") == 2  # both casts inline PTX, out of the reach of the folds


D = 1024


@tilefish.kernel(fp_fusion=False)
def axpb_strict(x: Tensor[R, D, f32], w: Tensor[D, f32], b: Tensor[D, f32]) -> Tensor[R, D, f32]:
    return x * w + b


@tilefish.kernel(fp_fusion=False)
def axpb_strict_bf16(x: Tensor[R, D, bf16], w: Tensor[D, bf16], b: Tensor[D, bf16]) -> Tensor[R, D, bf16]:
    return (x.to(f32) * w.to(f32) + b.to(f32)).to(bf16)


@tilefish.kernel
def axpb_fused(x: Tensor[R, D, f32], w: Tensor[D, f32], b: Tensor[D, f32]) -> Tensor[R, D, f32]:
    return x * w + b


def test_fp_fusion_false_rounds_twice_where_llvm_packs_the_lanes():
    g = torch.Generator(device=CUDA).manual_seed(7)
    x = torch.randn(4096, D, generator=g, device=CUDA)
    w, b = torch.randn(D, generator=g, device=CUDA), torch.randn(D, generator=g, device=CUDA)
    two = x * w + b  # torch: two kernels, two roundings
    one = (x.double() * w.double() + b.double()).float()  # the exact product and sum, rounded (twice, harmlessly for this check)
    assert not same_bytes(two, one)  # the data tells one rounding from two
    assert same_bytes(tilefish.run(axpb_strict, x, w, b), two)
    xb, wb, bb = x.to(torch.bfloat16), w.to(torch.bfloat16), b.to(torch.bfloat16)
    assert same_bytes(tilefish.run(axpb_strict_bf16, xb, wb, bb), (xb.float() * wb.float() + bb.float()).to(torch.bfloat16))
    for k in (axpb_strict, axpb_strict_bf16):  # and the check finds nothing fused in either binary
        low = tilefish.lower(k)
        (spec,) = compile_specs(low.extern)
        built = runtime.jit_built(runtime._target(low), spec, torch.cuda.current_device())
        v = _fpcontract.check(built.ptx, built.cubin, built.options, key=built.sha256)
        assert v is None or not v.contracted
        assert "f32x2" not in built.ptx
    tilefish.run(axpb_fused, x, w, b)  # the compiler's choice: one rounding or two, either is the contract
