"""Run the generated kernels on a pure-Python stand-in for Triton and compare with plain-Python references.

This checks the logic of the generated code (addressing, masks, reductions over padded lanes, which
value reaches which pointer) without a GPU. It is not a numerics test: see ``fake_triton`` for
what the stand-in does and does not model.
"""

import math
import random

import fake_triton as ft
import pytest
from conftest import make_my_kernels, readme_snippet
from lower_cases import gelu_exact, layernorm, softmax, swiglu

import tilefish
from tilefish import Dim, InOut, Out, Tensor, f32
from tilefish.aot import compile_specs

SENTINEL = 7.0  # a value no kernel here writes, to prove every element was written


@pytest.fixture(autouse=True)
def fake_triton(monkeypatch):
    ft.install(monkeypatch)


def rand_buf(rng, n, dtype, scale=3.0):
    return ft.Buf([rng.uniform(-scale, scale) for _ in range(n)], dtype)


def blank(n, dtype):
    return ft.Buf([SENTINEL] * n, dtype)


def run(lowered, buffers, dims, scalars=None, spec=None, source=None):
    """Launch one spec of ``lowered`` (its only one by default) with the grid its spec says."""
    spec = spec or compile_specs(lowered.extern)[0]
    kwargs = {}
    for i, (name, kind) in enumerate(spec.signature.items()):
        if kind == "constexpr":
            kwargs[name] = spec.constants[(i,)]
        elif kind == "i32":
            kwargs[name] = (scalars or {})[name]
        else:
            kwargs[name] = buffers[name]
    env = {"cdiv": lambda a, b: -(-a // b), **dims}
    (grid,) = eval(spec.grid, {"__builtins__": {}}, env)  # the generator's own grid string
    kernel = ft.load_kernel(source or lowered.source, lowered.fn_name)
    ft.launch(kernel, grid, **kwargs)


def close(got, want, rtol, atol=1e-6):
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want)):
        assert abs(g - w) <= atol + rtol * abs(w), f"element {i}: got {g}, want {w}"


def rows(flat, width):
    return [flat[i : i + width] for i in range(0, len(flat), width)]


# -- the README kernel: InOut store, rsqrt of a mean, one rounding point ---------------------------------------------


@pytest.fixture(scope="module")
def rmsnorm():
    ns = {"my_kernels": make_my_kernels()}
    exec(compile(readme_snippet(), "README.md", "exec"), ns)
    return ns["residual_add_rmsnorm"].lower()


def test_residual_add_rmsnorm(rmsnorm):
    rng = random.Random(1)
    R, D = 3, 2560
    x, residual, w = rand_buf(rng, R * D, ft.bfloat16), rand_buf(rng, R * D, ft.float32), rand_buf(rng, D, ft.bfloat16)
    r = [a + b for a, b in zip(residual.data, x.data)]
    want = []
    for row in rows(r, D):
        rstd = 1.0 / math.sqrt(sum(v * v for v in row) / D + 1e-6)
        want += [v * rstd * wv for v, wv in zip(row, w.data)]
    out = blank(R * D, ft.bfloat16)
    run(rmsnorm, {"x": x, "residual": residual, "w": w, "out": out}, {"M": R})
    close(residual.data, r, rtol=1e-6)  # the InOut parameter holds the final version
    close(out.data, want, rtol=1e-2)  # bf16 output
    assert all(ft._bf16(v) == v for v in out.data)  # and it is rounded to bf16


def test_layernorm_masks_padded_lanes_in_both_means():
    lowered = layernorm.lower()
    rng = random.Random(2)
    R, D = 4, 768  # 768 of 1024 lanes are real
    x, w, b = rand_buf(rng, R * D, ft.bfloat16), rand_buf(rng, D, ft.bfloat16), rand_buf(rng, D, ft.bfloat16)
    want = []
    for row in rows(x.data, D):
        mean = sum(row) / D
        var = sum((v - mean) ** 2 for v in row) / D
        rstd = 1.0 / math.sqrt(var + 1e-5)
        want += [(v - mean) * rstd * wv + bv for v, wv, bv in zip(row, w.data, b.data)]
    out = blank(R * D, ft.bfloat16)
    run(lowered, {"x": x, "w": w, "b": b, "out": out}, {"M": R})
    close(out.data, want, rtol=2e-2, atol=2e-2)


def test_softmax_over_a_row_that_is_not_a_power_of_two():
    lowered = softmax.lower()
    rng = random.Random(3)
    R, D = 3, 1000
    x = rand_buf(rng, R * D, ft.float32, scale=10.0)
    want = []
    for row in rows(x.data, D):
        top = max(row)
        e = [math.exp(v - top) for v in row]
        want += [v / sum(e) for v in e]
    out = blank(R * D, ft.float32)
    run(lowered, {"x": x, "out": out}, {"M": R})
    close(out.data, want, rtol=1e-6)
    for row in rows(out.data, D):
        assert abs(sum(row) - 1.0) < 1e-5


def test_the_stand_in_notices_a_missing_mask_and_a_missing_zeroing():
    """The checks above have teeth: break the generated source and they fail."""
    lowered = softmax.lower()
    rng = random.Random(3)
    R, D = 3, 1000
    x = rand_buf(rng, R * D, ft.float32, scale=0.5)  # small values: the 24 padded lanes weigh about as much as real ones
    unmasked = lowered.source.replace(", mask=mask", "")
    with pytest.raises(IndexError):
        run(lowered, {"x": x, "out": blank(R * D, ft.float32)}, {"M": R}, source=unmasked)
    unzeroed = lowered.source.replace("tl.where(mask, v3, 0.0)", "v3")
    assert unzeroed != lowered.source
    out = blank(R * D, ft.float32)
    run(lowered, {"x": x, "out": out}, {"M": R}, source=unzeroed)
    assert abs(sum(out.data[:D]) - 1.0) > 1e-3  # padded lanes leaked exp(0 - max) into the sum


# -- elementwise: a tail of lanes beyond n ---------------------------------------------------------------------------


def test_gelu_with_a_tail():
    lowered = gelu_exact.lower()
    rng = random.Random(4)
    T = 1504  # two programs of 1024 lanes, the second one mostly masked; divisible by 16
    x = rand_buf(rng, T, ft.bfloat16)
    want = [0.5 * v * (1.0 + math.erf(v * 0.7071067811865476)) for v in x.data]
    out = blank(T, ft.bfloat16)
    run(lowered, {"x": x, "out": out}, {"T": T}, scalars={"n": T})
    close(out.data, want, rtol=1e-2, atol=1e-3)


def test_swiglu_over_two_inputs():
    lowered = swiglu.lower()
    rng = random.Random(5)
    M, F = 2, 3072
    gate, up = rand_buf(rng, M * F, ft.bfloat16), rand_buf(rng, M * F, ft.bfloat16)
    want = [g / (1.0 + math.exp(-g)) * u for g, u in zip(gate.data, up.data)]
    out = blank(M * F, ft.bfloat16)
    run(lowered, {"gate": gate, "up": up, "out": out}, {"M": M}, scalars={"n": M * F})
    close(out.data, want, rtol=1e-2, atol=1e-3)


# -- addressing, per-row values, Out and InOut ---------------------------------------------------------------------


def test_per_row_results_and_inputs():
    Dn = Dim("Dn", 100)  # 100 of 128 lanes are real
    M = Dim("M")

    @tilefish.kernel
    def stats(x: Tensor[M, Dn, f32], s: Tensor[M, 1, f32]) -> tuple[Tensor[M, f32], Tensor[M, 1, f32], Tensor[M, Dn, f32]]:
        return tilefish.sum(x, axis=-1), tilefish.max(x, axis=-1, keepdims=True), x * s

    lowered = stats.lower()
    rng = random.Random(6)
    R, D = 5, 100
    x, s = rand_buf(rng, R * D, ft.float32), rand_buf(rng, R, ft.float32)
    sums, maxes, scaled = blank(R, ft.float32), blank(R, ft.float32), blank(R * D, ft.float32)
    run(lowered, {"x": x, "s": s, "out0": sums, "out1": maxes, "out2": scaled}, {"M": R})
    close(sums.data, [sum(row) for row in rows(x.data, D)], rtol=1e-6)
    close(maxes.data, [max(row) for row in rows(x.data, D)], rtol=1e-6)
    close(scaled.data, [v * s.data[i // D] for i, v in enumerate(x.data)], rtol=1e-6)


def test_out_and_inout_parameters():
    M = Dim("M")

    @tilefish.kernel
    def twice(x: InOut[Tensor[M, 6, f32]], y: Out[Tensor[M, 6, f32]]):
        a = x * 2.0
        x[...] = a  # superseded: only the last store reaches memory
        b = x + 1.0
        x[...] = b
        y[...] = a

    lowered = twice.lower()
    rng = random.Random(7)
    x = rand_buf(rng, 12, ft.float32)
    start = list(x.data)
    y = blank(12, ft.float32)
    run(lowered, {"x": x, "y": y}, {"M": 2}, scalars={"n": 12})
    close(x.data, [v * 2.0 + 1.0 for v in start], rtol=1e-6)
    close(y.data, [v * 2.0 for v in start], rtol=1e-6)


def test_every_bucket_of_a_bucketed_row_length_runs_with_its_own_constants():
    Hb = Dim("Hb", buckets=[96, 128, 200])
    M = Dim("M")

    @tilefish.kernel
    def rms(x: Tensor[M, Hb, f32]) -> Tensor[M, Hb, f32]:
        return x * tilefish.rsqrt(tilefish.mean(x * x, axis=-1, keepdims=True) + 1e-6)

    lowered = rms.lower()
    for spec, D in zip(compile_specs(lowered.extern), (96, 128, 200)):
        rng = random.Random(D)
        R = 2
        x = rand_buf(rng, R * D, ft.float32)
        want = []
        for row in rows(x.data, D):
            r = 1.0 / math.sqrt(sum(v * v for v in row) / D + 1e-6)
            want += [v * r for v in row]
        out = blank(R * D, ft.float32)
        run(lowered, {"x": x, "out": out}, {"M": R}, spec=spec)
        close(out.data, want, rtol=1e-6)


def test_a_returned_input_is_copied():
    M = Dim("M")

    @tilefish.kernel
    def identity(x: Tensor[M, 5, f32]) -> Tensor[M, 5, f32]:
        return x

    rng = random.Random(8)
    x = rand_buf(rng, 15, ft.float32)
    out = blank(15, ft.float32)
    run(identity.lower(), {"x": x, "out": out}, {"M": 3}, scalars={"n": 15})
    assert out.data == x.data


def test_parameters_named_like_the_modules_own_names_still_run():
    """``tl`` and ``float`` are what the generated code calls: renamed parameters must not shadow them."""
    M = Dim("M")

    @tilefish.kernel
    def clash(tl: Tensor[M, 7, f32], float: Tensor[M, 7, f32], triton: Tensor[M, 7, f32]) -> Tensor[M, 1, f32]:
        return tilefish.max(tl - float + triton, axis=-1, keepdims=True)  # float("-inf") masks the padded lanes

    lowered = clash.lower()
    rng = random.Random(9)
    a, b, c = (rand_buf(rng, 3 * 7, ft.float32) for _ in range(3))
    out = blank(3, ft.float32)
    run(lowered, {"tl_1": a, "float_1": b, "triton_1": c, "out": out}, {"M": 3})
    ra, rb, rc = rows(a.data, 7), rows(b.data, 7), rows(c.data, 7)
    want = [max(x - y + z for x, y, z in zip(ra[r], rb[r], rc[r])) for r in range(3)]
    close(out.data, want, rtol=1e-6)
