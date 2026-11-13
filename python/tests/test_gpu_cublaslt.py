"""The cuBLASLt operation of ``tilefish._cublaslt`` on a real GPU: ``D[M, N] = A[M, K] @ B[N, K]^T (+ bias[N])`` in bf16 with
an f32 accumulator and one rounding, run with a pinned algorithm.

* correctness against an f64 CPU reference over the shapes of a transformer's projections, with and without a bias;
* the translation to column-major, on small integer data whose every sum is exact in bf16 (no rounding to hide a swap);
* the pin: the same hex string and workspace when pinned twice, and in two other processes; the same output bytes on every run,
  also across two other processes, with other garbage in the output buffer, other workspace budgets and other streams;
* what is refused: a corrupted algorithm (every byte flipped in turn, and chosen corruptions), an algorithm for another
  problem, a workspace budget that is too small, misaligned pointers;
* threads pinning and running at once, sharing the one handle of the device;
* the ``"torch"`` posture: with torch's default reduction its result *is* ``torch.mm`` / ``torch.addmm``'s, byte for byte (asserted); an
  imported configuration is the heuristic's algorithm; a pin is the same in another process; a split-K that adds its partial sums in bf16
  runs only with ``allow_bf16_reduction``;
* two reports that only print (run with ``-s``): what the heuristic picked, and how often the pinned result is byte for byte
  what ``torch.mm`` / ``torch.addmm`` return.

Measured on a B300 MIG slice (sm_103, 30 SMs) with cuBLASLt 12.8.4 (``version() == 120804``): the heuristic picks algorithm 66
for every problem here (split-K, with f32 partial sums in the workspace, only for a small ``M * N`` and a very large ``K``).
``cublasLtMatmulAlgoCheck`` itself is a weak judge of an algorithm's bytes: it accepts a changed version field, changed
padding, a reserved CTA swizzling or an out-of-range custom option, and it crashes the process on a tile id of ``2**31`` or
more. ``tilefish._cublaslt`` therefore holds the string to what cuBLASLt itself would encode before it asks (see
``_validate_algo``); the corruption tests below are about that whole path.
"""

import functools
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import threading

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")

from gpu_util import DEVICE, bits, ordinal, randn  # noqa: E402

import tilefish  # noqa: E402
from tilefish import TilefishRuntimeError  # noqa: E402
from tilefish import _cublaslt as L  # noqa: E402

BF16 = torch.bfloat16
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True, scope="module")
def _cuda_is_up():
    torch.zeros(1, device=DEVICE)  # the context and torch's own cuBLASLt exist before anything is pinned


def current_stream():
    return torch.cuda.current_stream().cuda_stream


def run(a, b, bias=None, pin=None, *, out=None, stream=None, workspace_bytes=None):
    """``D = A @ B^T (+ bias)`` with the pinned algorithm (pinned here when ``pin`` is not given); returns ``D``."""
    m, k = a.shape
    n = b.shape[0]
    pin = pin or L.pin_algo(m, n, k, bias is not None)
    d = out if out is not None else torch.empty((m, n), dtype=BF16, device=DEVICE)
    L.matmul(
        a.data_ptr(), b.data_ptr(), d.data_ptr(), None if bias is None else bias.data_ptr(), m, n, k, pin.algo_hex,
        workspace_bytes=pin.workspace_required if workspace_bytes is None else workspace_bytes,
        stream=current_stream() if stream is None else stream,
    )  # fmt: skip
    return d


def inputs(m, n, k, with_bias, seed=0):
    a = randn((m, k), BF16, seed=seed)
    b = randn((n, k), BF16, seed=seed + 1)
    bias = randn((n,), BF16, seed=seed + 2) if with_bias else None
    return a, b, bias


def same(x, y):
    return x.shape == y.shape and torch.equal(bits(x), bits(y))


# -- the f64 reference -------------------------------------------------------------------------------------------------------


def round_to_bf16(x):
    """The nearest bf16 of an f64 tensor, ties to even, in one rounding (``x.to(bfloat16)`` would round to f32 first)."""
    sign = torch.where(x < 0, -1.0, 1.0).double()
    magnitude = x.abs().contiguous().view(torch.int64)
    magnitude = (magnitude + ((1 << 44) - 1) + ((magnitude >> 45) & 1)) & ~((1 << 45) - 1)  # keep 7 of the 52 mantissa bits
    return (magnitude.view(torch.float64) * sign).to(BF16)


def spacing(x):
    """The distance of a bf16 value (as f64) to the next one of the same binade; 0 for zero."""
    _, exponent = torch.frexp(x.double())
    return torch.where(x == 0, 0.0, torch.pow(2.0, (exponent - 8).double()))


def f64_report(out, a, b, bias):
    """Compare a bf16 result with ``round_to_bf16(A.double() @ B.double().T + bias)``.

    Returns ``(differing, max_ulp, worst)``: the number of elements that are not that rounding, the largest distance of those in
    bf16 values, and the largest ``|out - exact| / (half a bf16 spacing + slack)`` (at most 1 when every element is correct).
    The ``slack`` is what any f32 accumulation order may be off by, ``gamma_(K+1) * (|A| @ |B|^T + |bias|)`` with
    ``gamma_n = n * 2^-24 / (1 - n * 2^-24)``: an element may differ from the rounding of the exact value only when the exact
    value is within the slack of a rounding boundary, which is where a difference of the f32 sums is unavoidable.
    """
    a64, b64 = a.cpu().double(), b.cpu().double()
    exact = a64 @ b64.T
    absolute = a64.abs() @ b64.abs().T
    if bias is not None:
        exact = exact + bias.cpu().double()
        absolute = absolute + bias.cpu().double().abs()
    reference = round_to_bf16(exact)
    actual = out.cpu()
    assert actual.shape == reference.shape and actual.dtype == BF16
    assert torch.isfinite(actual.double()).all()
    k = a.shape[1] + 1
    gamma = k * 2.0**-24 / (1 - k * 2.0**-24)
    allowed = 0.5 * spacing(actual) + gamma * absolute
    error = (actual.double() - exact).abs()
    differing = ~torch.eq(bits(actual).reshape(-1, 2), bits(reference).reshape(-1, 2)).all(-1).reshape(actual.shape)
    max_ulp = int((ordinal(actual) - ordinal(reference)).abs().max())
    worst = float((error / allowed.clamp_min(1e-300)).max())
    return int(differing.sum()), max_ulp, worst


# -- correctness -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("with_bias", [False, True], ids=["plain", "bias"])
@pytest.mark.parametrize("k", [1024, 3072, 4096])
@pytest.mark.parametrize("n", [1024, 3072])
@pytest.mark.parametrize("m", [32, 64, 320])
def test_matches_an_f64_reference_up_to_the_f32_accumulation(m, n, k, with_bias):
    a, b, bias = inputs(m, n, k, with_bias, seed=m + n + k)
    out = run(a, b, bias)
    torch.cuda.synchronize()
    differing, max_ulp, worst = f64_report(out, a, b, bias)
    print(f"F64 m={m} n={n} k={k} bias={with_bias}: {differing}/{m * n} elements are not the correctly rounded f64 result "
          f"(max {max_ulp} bf16 ulp, near zero where the sum cancels), worst |err|/(half ulp + f32 slack) = {worst:.3g}")  # fmt: skip
    assert worst <= 1.0, "an element is off by more than a half bf16 ulp plus what f32 accumulation can explain"
    assert differing <= 0.01 * m * n  # a systematic rounding error (a second rounding, say) would show here


def exact_integers(m, n, k, with_bias):
    """Small integer data whose every product and sum is exact in bf16 and f32: the result is one right answer."""
    gen = torch.Generator().manual_seed(m * 10_000 + n * 100 + k)
    a = torch.randint(-1, 2, (m, k), generator=gen).to(BF16)
    b = torch.randint(-1, 2, (n, k), generator=gen).to(BF16)
    bias = torch.randint(-20, 21, (n,), generator=gen).to(BF16) if with_bias else None
    return a, b, bias


@pytest.mark.parametrize("with_bias", [False, True], ids=["plain", "bias"])
@pytest.mark.parametrize(
    "m,n,k", [(40, 24, 16), (8, 128, 32), (200, 56, 72), (1, 64, 8), (3, 8, 8), (96, 16, 128), (16, 200, 40), (17, 33, 64)]
)
def test_translation_to_column_major_on_exact_integers(m, n, k, with_bias):
    """``|sum| <= k <= 128`` and a bias up to 20 stay below 256, where every integer is a bf16: any swapped operand, wrong
    transpose or bias along the wrong axis gives other numbers, and any rounding at all would too."""
    a, b, bias = exact_integers(m, n, k, with_bias)
    expected = a.double() @ b.double().T
    if with_bias:
        expected = expected + bias.double()
    out = run(a.to(DEVICE), b.to(DEVICE), None if bias is None else bias.to(DEVICE))
    torch.cuda.synchronize()
    assert torch.equal(out.cpu().double(), expected), f"{int((out.cpu().double() != expected).sum())} of {m * n} elements differ"


@pytest.mark.parametrize("with_bias", [False, True], ids=["plain", "bias"])
@pytest.mark.parametrize("m,n,k", [(5, 70, 100), (1, 1, 1), (7, 9, 33), (130, 250, 18)])
def test_sizes_that_are_not_multiples_of_8(m, n, k, with_bias):
    """The rows are then not 16-byte aligned: the heuristic picks an algorithm for the leading dimensions that there are."""
    a, b, bias = exact_integers(m, n, k, with_bias)
    expected = a.double() @ b.double().T
    if with_bias:
        expected = expected + bias.double()
    out = run(a.to(DEVICE), b.to(DEVICE), None if bias is None else bias.to(DEVICE))
    torch.cuda.synchronize()
    assert torch.equal(out.cpu().double(), expected)


def test_the_bias_is_added_before_the_one_rounding():
    """``A @ B^T = 1 + 2**-8`` is halfway between the bf16 values ``1`` and ``1 + 2**-7`` (it rounds to ``1``, to even), and
    the bias is ``2**-8``. Added in f32 before the one rounding the sum is ``1 + 2**-7``, a bf16 value; rounding the product
    first and adding the bias after gives ``1`` again (``1 + 2**-8`` is halfway once more)."""
    a = torch.zeros((8, 8), dtype=BF16)
    b = torch.zeros((8, 8), dtype=BF16)
    a[:, 0], a[:, 1] = 1.0, 2.0**-8
    b[:, 0], b[:, 1] = 1.0, 1.0  # the products 1 * 1 and 2**-8 * 1
    bias = torch.full((8,), 2.0**-8, dtype=BF16)
    single = 1.0 + 2.0**-7
    assert float(torch.tensor(1.0 + 2.0**-8).to(BF16)) == 1.0 and float(torch.tensor(1.0 + 2.0**-8).to(BF16)) + 2.0**-8 != single
    out = run(a.to(DEVICE), b.to(DEVICE), bias.to(DEVICE))
    plain = run(a.to(DEVICE), b.to(DEVICE))
    torch.cuda.synchronize()
    assert torch.equal(out.cpu().double(), torch.full((8, 8), single, dtype=torch.float64))
    assert torch.equal(plain.cpu().double(), torch.full((8, 8), 1.0, dtype=torch.float64))  # without the bias: the product rounds to 1


# -- the pin -----------------------------------------------------------------------------------------------------------------

PIN_SHAPES = [(32, 1024, 1024), (64, 3072, 1024), (320, 1024, 3072), (64, 4096, 3072), (320, 3072, 4096), (1, 128, 28672)]


def test_the_library_is_the_one_torch_has_mapped_and_it_has_a_version():
    path = L.library_path()
    assert os.path.basename(path).startswith("libcublasLt.so")
    mapped = L._mapped_paths()
    assert mapped == [os.path.realpath(path)], "exactly one libcublasLt is mapped, and it is the one the binding uses"
    version = L.version()
    assert isinstance(version, int) and version >= 120000
    print(f"CUBLASLT version={version} library={path} device={torch.cuda.get_device_name()} "
          f"sms={torch.cuda.get_device_properties(0).multi_processor_count}")  # fmt: skip


@pytest.mark.parametrize("with_bias", [False, True], ids=["plain", "bias"])
@pytest.mark.parametrize("m,n,k", PIN_SHAPES)
def test_pinning_twice_gives_one_pin(m, n, k, with_bias):
    first = L.pin_algo(m, n, k, with_bias)
    second = L.pin_algo(m, n, k, with_bias)
    assert first == second
    assert len(first.algo_hex) == 128 and first.algo_hex == first.algo_hex.lower()
    assert first.workspace_limit == L.DEFAULT_WORKSPACE_BYTES
    assert 0 <= first.workspace_required <= first.workspace_limit
    assert L.check_algo(m, n, k, with_bias, first.algo_hex, workspace_bytes=first.workspace_required) == first.workspace_required
    assert L.check_algo(m, n, k, with_bias, first.algo_hex, workspace_bytes=first.workspace_limit, device_index=0) == first.workspace_required
    info = L.algo_info(first.algo_hex)
    assert info["reduction_scheme"] in L.ALLOWED_REDUCTION_SCHEMES and info["splitk_num"] >= 1
    if info["splitk_num"] > 1:
        assert first.workspace_required > 0  # split-K keeps its partial sums in the workspace


def test_every_pin_over_a_grid_of_sizes_passes_its_own_check_and_keeps_the_one_rounding():
    """The heuristic is asked for split-K with f32 partial sums at most; whatever it returns for tiny, odd, tall and wide
    problems must be an algorithm that tilefish's own check lets through, with a workspace inside the limit."""
    count, ids, none = 0, {}, []
    for m in (1, 7, 64, 320, 4096):
        for n in (1, 70, 1024, 3070):
            for k in (1, 18, 1000, 4096, 28672):
                for with_bias in (False, True):
                    try:
                        pin = L.pin_algo(m, n, k, with_bias)
                    except TilefishRuntimeError as e:
                        # a GEMM with a bias is pinned without split-K, and for a single column of a very long K the library may have none
                        # (an RTX 5090: 64x1x28672 and 320x1x28672); said so, and nothing is pinned
                        assert with_bias and n == 1 and "asked for one without split-K" in str(e), str(e)
                        none.append((m, n, k))
                        continue
                    info = L.algo_info(pin.algo_hex)
                    assert info["reduction_scheme"] in L.ALLOWED_REDUCTION_SCHEMES
                    assert (info["splitk_num"] > 1) == (info["reduction_scheme"] != L.CUBLASLT_REDUCTION_SCHEME_NONE)
                    assert pin.workspace_required <= pin.workspace_limit
                    assert L.check_algo(m, n, k, with_bias, pin.algo_hex, workspace_bytes=pin.workspace_required) == pin.workspace_required
                    ids[info["id"]] = ids.get(info["id"], 0) + 1
                    count += 1
    print(f"GRID {count} pins, algorithm ids {dict(sorted(ids.items()))}; no algorithm without split-K for the biased {none}")
    assert len(none) <= 4


def test_a_workspace_limit_is_a_parameter():
    default = L.pin_algo(1, 128, 28672)
    none = L.pin_algo(1, 128, 28672, workspace_bytes=0)
    small = L.pin_algo(1, 128, 28672, workspace_bytes=1024)
    assert none.workspace_limit == 0 and none.workspace_required == 0
    assert small.workspace_limit == 1024 and small.workspace_required <= 1024
    assert default.workspace_limit == L.DEFAULT_WORKSPACE_BYTES
    print(f"WORKSPACE 1x128x28672: default limit needs {default.workspace_required}, limit 1024 needs {small.workspace_required}, "
          f"limit 0 needs {none.workspace_required}; splitk {L.algo_info(default.algo_hex)['splitk_num']} / "
          f"{L.algo_info(small.algo_hex)['splitk_num']} / {L.algo_info(none.algo_hex)['splitk_num']}")  # fmt: skip


def payload():
    """What two processes must agree on: the library, its version, the pin of each problem and a digest of each pinned result."""
    result = {"version": L.version(), "library": L.library_path(), "pins": {}, "digests": {}}
    for m, n, k in PIN_SHAPES:
        for with_bias in (False, True):
            key = f"{m}x{n}x{k}-{'bias' if with_bias else 'plain'}"
            pin = L.pin_algo(m, n, k, with_bias)
            result["pins"][key] = [pin.algo_hex, pin.workspace_required, pin.workspace_limit]
            a, b, bias = inputs(m, n, k, with_bias, seed=21)
            out = run(a, b, bias, pin)
            torch.cuda.synchronize()
            result["digests"][key] = hashlib.sha256(bits(out).cpu().numpy().tobytes()).hexdigest()
    return result


def other_process_payload():
    env = dict(os.environ)
    src = os.path.dirname(os.path.dirname(os.path.abspath(tilefish.__file__)))
    env["PYTHONPATH"] = os.pathsep.join(p for p in [src, TESTS_DIR, env.get("PYTHONPATH", "")] if p)
    code = "import json, test_gpu_cublaslt as t; print('PAYLOAD ' + json.dumps(t.payload()))"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=900)
    assert done.returncode == 0, done.stderr[-3000:]
    (line,) = [line for line in done.stdout.splitlines() if line.startswith("PAYLOAD ")]
    return json.loads(line[len("PAYLOAD "):])


def test_pins_and_results_are_the_same_in_two_other_processes():
    here = payload()
    first = other_process_payload()
    second = other_process_payload()
    assert first["version"] == second["version"] == here["version"]
    assert first["library"] == second["library"] == here["library"]
    assert first["pins"] == second["pins"] == here["pins"]
    assert first["digests"] == second["digests"] == here["digests"]
    print(f"PROCESSES {len(here['pins'])} pins and {len(here['digests'])} output digests identical in this and two other processes")


@pytest.mark.parametrize("with_bias", [False, True], ids=["plain", "bias"])
@pytest.mark.parametrize("m,n,k", [(32, 1024, 1024), (320, 3072, 4096), (1, 128, 28672)])
def test_the_result_is_byte_identical_on_every_run(m, n, k, with_bias):
    a, b, bias = inputs(m, n, k, with_bias, seed=5)
    pin = L.pin_algo(m, n, k, with_bias)
    first = run(a, b, bias, pin).clone()
    out = torch.empty((m, n), dtype=BF16, device=DEVICE)
    for fill in (float("nan"), 0.0, 1.0, -3.5):  # whatever the output buffer held, it is overwritten, not accumulated into
        out.fill_(fill)
        run(a, b, bias, pin, out=out)
        assert same(out, first), f"differs with the output buffer filled with {fill}"
    for _ in range(5):
        assert same(run(a, b, bias, pin), first)
    assert same(run(a, b, bias), first)  # pinned again: the same algorithm
    assert same(run(a, b, bias, pin, workspace_bytes=pin.workspace_limit), first)  # a larger workspace budget changes nothing
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        other = run(a, b, bias, pin, stream=stream.cuda_stream)
    stream.synchronize()
    assert same(other, first)


@pytest.mark.parametrize("with_bias", [False, True], ids=["plain", "bias"])
def test_pointers_that_are_16_byte_but_not_256_byte_aligned_give_the_same_bytes(with_bias):
    m, n, k = 64, 1024, 1024
    a, b, bias = inputs(m, n, k, with_bias, seed=9)
    pin = L.pin_algo(m, n, k, with_bias)
    reference = run(a, b, bias, pin).clone()

    def shifted(t, elements):  # the same values, at an address that is 16 (and no more) bytes beyond a 512-byte boundary
        flat = torch.zeros(t.numel() + 64, dtype=BF16, device=DEVICE)
        view = flat[elements : elements + t.numel()].view(t.shape)
        view.copy_(t)
        assert view.data_ptr() % 16 == 0
        return view

    for elements in (8, 24, 40):
        d = torch.zeros(m * n + 64, dtype=BF16, device=DEVICE)[elements : elements + m * n].view(m, n)
        assert d.data_ptr() % 256 != 0
        run(shifted(a, elements), shifted(b, elements), None if bias is None else shifted(bias, elements), pin, out=d)
        torch.cuda.synchronize()
        assert same(d, reference), f"differs at an offset of {elements * 2} bytes"


# -- the algorithm with a workspace (split-K) ----------------------------------------------------------------------------------


@pytest.mark.parametrize("with_bias", [False, True], ids=["plain", "bias"])
@pytest.mark.parametrize("m,n,k", [(1, 128, 28672), (8, 1024, 16384), (1, 1024, 28672)])
def test_an_algorithm_that_needs_a_workspace(m, n, k, with_bias):
    pin = L.pin_algo(m, n, k, with_bias)
    if with_bias:  # a GEMM with a bias is pinned without split-K unless it opts in to bf16 reductions (the bias epilogue may round twice)
        assert L.algo_info(pin.algo_hex)["splitk_num"] == 1
        a, b, bias = inputs(m, n, k, True, seed=3)
        assert f64_report(run(a, b, bias, pin), a, b, bias)[2] <= 1.0
        return
    if pin.workspace_required == 0:
        pytest.skip(f"the heuristic picks an algorithm without a workspace for {m}x{n}x{k} here")
    info = L.algo_info(pin.algo_hex)
    print(f"SPLITK m={m} n={n} k={k} bias={with_bias}: workspace {pin.workspace_required}, {info}")
    assert info["splitk_num"] > 1 and info["reduction_scheme"] == L.CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE
    a, b, bias = inputs(m, n, k, with_bias, seed=3)
    first = run(a, b, bias, pin).clone()
    torch.cuda.synchronize()
    differing, max_ulp, worst = f64_report(first, a, b, bias)
    assert worst <= 1.0, f"max ulp {max_ulp}, worst {worst}"
    for _ in range(3):
        assert same(run(a, b, bias, pin), first)
    # the workspace is taken from torch's caching allocator on the stream of the call and freed after it: nothing leaks
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(10):
        run(a, b, bias, pin)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    with pytest.raises(TilefishRuntimeError, match=f"needs {pin.workspace_required} bytes of workspace.*more than the {pin.workspace_required - 1} recorded"):
        L.check_algo(m, n, k, with_bias, pin.algo_hex, workspace_bytes=pin.workspace_required - 1)
    sentinel = torch.full((m, n), 7.0, dtype=BF16, device=DEVICE)
    with pytest.raises(TilefishRuntimeError, match="more than the 0 recorded"):
        run(a, b, bias, pin, out=sentinel, workspace_bytes=0)
    torch.cuda.synchronize()
    assert bool((sentinel == 7.0).all()), "a refused call must not have launched"


# -- what is refused ---------------------------------------------------------------------------------------------------------

# Where the fields sit in the 64 bytes of cublasLtMatmulAlgo_t in cuBLASLt 12.8.4 (found by flipping each byte in turn and
# reading the configuration back with cublasLtMatmulAlgoConfigGetAttribute); a test that relies on one checks it first.
FIELD_AT = {
    "id": (0, 4), "tile_id": (4, 4), "stages_id": (8, 4), "splitk_num": (12, 4), "reduction_scheme": (16, 4),
    "cta_swizzling": (20, 4), "custom_option": (24, 4), "inner_shape_id": (28, 2), "cluster_shape_id": (30, 2),
}  # fmt: skip

GOOD = (64, 3072, 1024)


def good_pin():
    return L.pin_algo(*GOOD, False)


def with_field(raw, name, value):
    start, width = FIELD_AT[name]
    changed = bytearray(raw)
    changed[start : start + width] = value.to_bytes(width, "little")
    if L.algo_info(bytes(changed).hex())[name] != value:
        pytest.skip(f"the algorithm layout of this cuBLASLt differs from the one this test was written against ({name})")
    return bytes(changed)


@functools.lru_cache(maxsize=None)
def cached_inputs(m, n, k, with_bias):
    return inputs(m, n, k, with_bias, seed=1)


def refused(algo_hex, *, shape=GOOD, bias=False, workspace_bytes=L.DEFAULT_WORKSPACE_BYTES):
    """The error that ``check_algo`` and ``matmul`` both raise for this algorithm (asserting they do), or ``None``."""
    m, n, k = shape
    a, b, bias_t = cached_inputs(m, n, k, bias)
    sentinel = torch.full((m, n), 7.0, dtype=BF16, device=DEVICE)
    errors = []
    for call in (
        lambda: L.check_algo(m, n, k, bias, algo_hex, workspace_bytes=workspace_bytes),
        lambda: L.matmul(
            a.data_ptr(), b.data_ptr(), sentinel.data_ptr(), None if bias_t is None else bias_t.data_ptr(), m, n, k, algo_hex,
            workspace_bytes=workspace_bytes, stream=current_stream(),
        ),
    ):  # fmt: skip
        try:
            call()
            errors.append(None)
        except TilefishRuntimeError as e:
            errors.append(e)
    torch.cuda.synchronize()
    assert (errors[0] is None) == (errors[1] is None), "check_algo and matmul disagree"
    if errors[0] is not None:
        assert bool((sentinel == 7.0).all()), "a refused matmul must not have launched"
    return errors[0]


def test_the_good_pin_is_accepted_by_both():
    assert refused(good_pin().algo_hex) is None


@pytest.mark.parametrize("name", ["all zeros", "all ones"])
def test_garbage_is_refused(name):
    error = refused(("00" if name == "all zeros" else "ff") * 64)
    assert error is not None and "is not one cuBLASLt produces" in str(error)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("tile_id", 2**31 + 164, "tile id"),  # cublasLtMatmulAlgoCheck crashes the process (SIGSEGV) on this one
        ("tile_id", 0xFFFFFFFF, "tile id"),
        ("tile_id", 0xFF0000A4, "tile id"),
        ("stages_id", 0xFFFFFFFF, "stages id"),
        ("splitk_num", 254, "split-K count 254"),  # cublasLtMatmulAlgoCheck takes it, with a workspace of 199753744 bytes
        ("splitk_num", 0, "split-K count"),
        ("reduction_scheme", L.CUBLASLT_REDUCTION_SCHEME_INPLACE, "reduction scheme"),
        ("reduction_scheme", L.CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE, "reduction scheme"),
        ("reduction_scheme", 255, "reduction scheme"),
        ("cta_swizzling", 255, "CTA swizzling"),  # accepted by cublasLtMatmulAlgoCheck
        ("cta_swizzling", 2**31, "CTA swizzling"),  # accepted by cublasLtMatmulAlgoCheck
        ("custom_option", 2**31, "custom option"),  # accepted by cublasLtMatmulAlgoCheck
        ("custom_option", 65536, "custom option"),
        ("id", 0x7FFFFFFF, "unknown"),
    ],
)
def test_a_corrupted_field_is_refused(field, value, reason):
    changed = with_field(bytes.fromhex(good_pin().algo_hex), field, value)
    error = refused(changed.hex())
    assert error is not None
    assert error.__class__ is TilefishRuntimeError
    assert error.args and any(word in str(error) for word in reason.split("|")), str(error)


@pytest.mark.parametrize("position", [32, 33, 34, 35, 36, 40, 44, 48, 52, 56, 60, 61, 62, 63])
def test_a_flipped_byte_of_the_version_the_types_or_the_padding_is_refused(position):
    """Bytes 32..63 hold the cuBLASLt version, the data and compute types and padding: cublasLtMatmulAlgoCheck accepts a
    change in the version and in the padding (measured); tilefish refuses, since cuBLASLt would not have encoded it."""
    raw = bytearray.fromhex(good_pin().algo_hex)
    for xor in (0x01, 0x80, 0xFF):
        changed = bytearray(raw)
        changed[position] ^= xor
        error = refused(bytes(changed).hex())
        assert error is not None, f"byte {position} ^ {xor:#x} was accepted"
        assert "is not one cuBLASLt produces" in str(error) or "not valid for" in str(error)


def test_every_flipped_byte_is_refused_or_is_another_valid_configuration():
    """Flip each of the 64 bytes in turn (three ways each). What cuBLASLt's own check says is not what is asserted; what tilefish
    lets through is: nothing but a different tile, cluster shape or inner shape (the last measured on an RTX 5090), which is a valid,
    different algorithm (it cannot be told from the pin by any check of the bytes; the manifest's digest is what holds the string)."""
    raw = bytes.fromhex(good_pin().algo_hex)
    original = L.algo_info(raw.hex())
    accepted, refused_count = [], 0
    for position in range(64):
        for xor in (0x01, 0x80, 0xFF):
            changed = bytearray(raw)
            changed[position] ^= xor
            error = refused(bytes(changed).hex())
            if error is None:
                info = L.algo_info(bytes(changed).hex())
                difference = {key for key in info if info[key] != original[key]}
                assert difference <= {"tile_id", "cluster_shape_id", "inner_shape_id"}, f"byte {position} ^ {xor:#x} was accepted with {difference} changed"
                accepted.append((position, xor, tuple(sorted(difference))))
            else:
                assert error.__class__ is TilefishRuntimeError
                refused_count += 1
    print(f"FLIPS refused {refused_count} of 192 single-byte corruptions; accepted {len(accepted)}: {accepted}")
    assert refused_count >= 150


def test_an_algorithm_that_is_valid_but_for_another_problem_is_refused_where_cublaslt_says_so():
    """The pin of 64x3072x1024 on 64x3070x1024: the rows of 3070 bf16 are not 16-byte aligned, which this algorithm needs
    (cublasLtMatmulAlgoCheck says CUBLAS_STATUS_NOT_SUPPORTED, measured)."""
    error = refused(good_pin().algo_hex, shape=(64, 3070, 1024))
    assert error is not None and "not valid for the matmul m=64 n=3070 k=1024" in str(error) and "CUBLAS_STATUS_NOT_SUPPORTED" in str(error)


def test_an_algorithm_for_another_size_that_cublaslt_accepts_runs_and_is_correct():
    """cublasLtMatmulAlgoCheck accepts most algorithms for most sizes (measured): a pin is the algorithm and the problem it was made
    for, and it is the caller's job (the manifest's) to keep them together. What is accepted must at least be right."""
    pin = L.pin_algo(320, 3072, 4096, True)
    a, b, bias = inputs(32, 1024, 1024, True, seed=4)
    need = L.check_algo(32, 1024, 1024, True, pin.algo_hex, workspace_bytes=L.DEFAULT_WORKSPACE_BYTES)  # (0 on a B300; a split-K's elsewhere)
    out = run(a, b, bias, pin, workspace_bytes=max(need, pin.workspace_required))
    torch.cuda.synchronize()
    _, max_ulp, worst = f64_report(out, a, b, bias)
    assert worst <= 1.0


@pytest.mark.parametrize("which", ["A", "B", "D", "bias"])
@pytest.mark.parametrize("bytes_off", [2, 4, 8])
def test_a_misaligned_pointer_is_refused_and_nothing_is_launched(which, bytes_off):
    m, n, k = 16, 64, 128
    a, b, bias = inputs(m, n, k, True)
    pin = L.pin_algo(m, n, k, True)
    sentinel = torch.full((m * n + 64,), 7.0, dtype=BF16, device=DEVICE)
    pointers = {"A": a.data_ptr(), "B": b.data_ptr(), "D": sentinel.data_ptr(), "bias": bias.data_ptr()}
    pointers[which] += bytes_off
    with pytest.raises(TilefishRuntimeError, match=rf"the {which} pointer 0x[0-9a-f]+ is not 16-byte aligned"):
        L.matmul(pointers["A"], pointers["B"], pointers["D"], pointers["bias"], m, n, k, pin.algo_hex, workspace_bytes=pin.workspace_required, stream=current_stream())
    torch.cuda.synchronize()
    assert bool((sentinel == 7.0).all())


def test_a_null_pointer_is_refused():
    a, b, bias = inputs(8, 16, 32, False)
    pin = L.pin_algo(8, 16, 32)
    with pytest.raises(TilefishRuntimeError, match="A pointer"):
        L.matmul(0, b.data_ptr(), b.data_ptr(), None, 8, 16, 32, pin.algo_hex, workspace_bytes=0, stream=current_stream())


def test_a_device_index_that_does_not_exist_is_refused():
    with pytest.raises(TilefishRuntimeError, match="out of range"):
        L.pin_algo(8, 16, 32, device_index=torch.cuda.device_count())


# -- threads and the handle --------------------------------------------------------------------------------------------------


def test_the_handle_is_created_once_per_device_however_many_threads_ask():
    L._destroy_handles()
    lib = L._lib()
    barrier = threading.Barrier(8)
    seen = []

    def ask():
        barrier.wait()
        seen.append(L._handle(lib, 0).value)

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == 8 and len(set(seen)) == 1 and seen[0]
    assert list(L._handles) == [0]


def test_threads_pin_and_run_at_once_on_streams_of_their_own():
    shapes = [(32, 1024, 1024), (64, 3072, 1024), (320, 1024, 3072), (1, 128, 28672)]
    data = {s: inputs(*s, True, seed=sum(s)) for s in shapes}
    expected = {s: run(*data[s]).clone() for s in shapes}
    pins = {s: L.pin_algo(*s, True) for s in shapes}
    torch.cuda.synchronize()
    errors = []
    barrier = threading.Barrier(6)

    def work(index):
        try:
            stream = torch.cuda.Stream()
            barrier.wait()
            for round_ in range(6):
                shape = shapes[(index + round_) % len(shapes)]
                with torch.cuda.stream(stream):
                    assert L.pin_algo(*shape, True) == pins[shape]
                    out = run(*data[shape], pins[shape], stream=stream.cuda_stream)
                stream.synchronize()
                assert same(out, expected[shape]), f"thread {index} round {round_} shape {shape} differs"
                L.check_algo(*shape, True, pins[shape].algo_hex, workspace_bytes=pins[shape].workspace_limit)
        except BaseException as e:  # noqa: BLE001 - reported by the main thread
            errors.append(f"thread {index}: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


# -- reports (nothing is asserted about the numbers) ---------------------------------------------------------------------------

TORCH_MN_K = [(1024, 1024), (3072, 1024), (1024, 3072), (4096, 3072), (3072, 4096)]


def test_report_what_the_heuristic_picks():
    lines = []
    for with_bias in (False, True):
        for m in (32, 64, 320):
            for n, k in TORCH_MN_K:
                pin = L.pin_algo(m, n, k, with_bias)
                info = L.algo_info(pin.algo_hex)
                lines.append(
                    f"PIN m={m} n={n} k={k} bias={with_bias}: id={info['id']} tile={info['tile_id']} stages={info['stages_id']} "
                    f"splitk={info['splitk_num']} reduction={info['reduction_scheme']} custom={info['custom_option']} "
                    f"cluster={info['cluster_shape_id']} workspace={pin.workspace_required}"
                )
    print("\n" + "\n".join(lines))
    assert len(lines) == 30


def test_report_how_often_the_pinned_result_is_torchs():
    """For each shape: is the pinned cuBLASLt result byte for byte ``torch.mm(a, b.t())`` (``torch.addmm(bias, a, b.t())``)?
    Prints the counts and the largest bf16-ulp distance of those that differ; asserts nothing about the rate."""
    for with_bias in (False, True):
        matched, total, worst_ulp, notes = 0, 0, 0, []
        for m in (32, 64, 320):
            for n, k in TORCH_MN_K:
                a, b, bias = inputs(m, n, k, with_bias, seed=m + n + k)
                ours = run(a, b, bias)
                theirs = torch.addmm(bias, a, b.t()) if with_bias else torch.mm(a, b.t())
                torch.cuda.synchronize()
                assert ours.shape == theirs.shape and ours.dtype == theirs.dtype
                total += 1
                if same(ours, theirs):
                    matched += 1
                else:
                    ulp = (ordinal(ours.cpu()) - ordinal(theirs.cpu())).abs()
                    worst_ulp = max(worst_ulp, int(ulp.max()))
                    notes.append(f"{m}x{n}x{k}: {int((ulp > 0).sum())} elements differ, max {int(ulp.max())} ulp")
        print(f"TORCH-MATCH {'torch.addmm' if with_bias else 'torch.mm'} (bias={with_bias}): {matched}/{total} shapes byte for byte; "
              f"max ulp when they differ: {worst_ulp if notes else 'n/a (none differ)'}" + (f"; {notes}" if notes else ""))  # fmt: skip


# -- the "torch" posture, imported configurations and bf16 partial sums ----------------------------------------------------------------------

# (M, N, K): transformer projections at several M, and the shapes where a split-K is chosen (a small M * N and a large K)
TORCH_SHAPES = [(m, n, k) for m in (1, 32, 64, 320, 1000, 2048) for n, k in TORCH_MN_K + [(1024, 4096), (1024, 7680), (5000, 1024)]]


def torch_pin(m, n, k, with_bias):
    """The "torch" posture with torch's default reduction (bf16 partial sums allowed) and torch's workspace limit for the op."""
    return L.pin_algo(
        m, n, k, with_bias, heuristic="torch", allow_bf16_reduction=True,
        workspace_bytes=L.TORCH_LT_WORKSPACE_BYTES if with_bias else L.DEFAULT_WORKSPACE_BYTES,
    )  # fmt: skip


def run_allowing_bf16(a, b, bias, pin):
    m, k = a.shape
    n = b.shape[0]
    d = torch.empty((m, n), dtype=BF16, device=DEVICE)
    L.matmul(a.data_ptr(), b.data_ptr(), d.data_ptr(), None if bias is None else bias.data_ptr(), m, n, k, pin.algo_hex,
             workspace_bytes=pin.workspace_limit, stream=current_stream(), allow_bf16_reduction=True)  # fmt: skip
    return d


def special(x, kind, seed):
    """``x`` (normal values) made into values of every exponent (``"wide"``: times 2**-20 .. 2**19) or into exact powers of two with 3% of them
    +-2**40 (``"traps"``), on the CPU (torch's jit-compiled kernels for such ops do not build for every architecture)."""
    g = torch.Generator().manual_seed(seed)
    v = x.float().cpu()
    sign = torch.where(v < 0, -1.0, 1.0)
    if kind == "wide":
        v = v * torch.pow(2.0, torch.randint(-20, 20, v.shape, generator=g).float())
    else:
        v = sign * torch.pow(2.0, torch.randint(-8, 9, v.shape, generator=g).float())
        v = torch.where(torch.rand(v.shape, generator=g) < 0.03, sign * 2.0**40, v)
    return v.to(BF16).to(DEVICE)


@pytest.mark.parametrize("with_bias", [False, True])
def test_the_torch_posture_pins_what_torch_runs(with_bias):
    """The README's claim: under ``heuristic="torch"`` (with torch's default reduction) the pinned result is ``torch.addmm(bias, a, b.t())``
    (``torch.mm(a, b.t())`` without a bias) byte for byte. ``addmm`` asks cuBLASLt's heuristic as this posture does; ``mm`` runs
    ``cublasGemmEx``, whose choice the posture reproduces with torch's cuBLAS workspace. Measured, on the shapes of a transformer's
    projections and of a split-K; the 3 operand sets of each are normal values, values of every exponent, and exact powers of two with
    huge ones that cancel (so a change of the summation order shows)."""
    misses = []
    for m, n, k in TORCH_SHAPES:
        pin = torch_pin(m, n, k, with_bias)
        for seed, kind in enumerate(("normal", "wide", "traps")):
            a, b, bias = inputs(m, n, k, with_bias, seed=seed * 7 + m)
            if kind != "normal":
                a, b = (special(x, kind, seed * 7 + m + i) for i, x in enumerate((a, b)))
            ours = run_allowing_bf16(a, b, bias, pin)
            theirs = torch.addmm(bias, a, b.t()) if with_bias else torch.mm(a, b.t())
            torch.cuda.synchronize()
            if not same(ours, theirs):
                misses.append((m, n, k, kind, L.algo_info(pin.algo_hex)))
    assert not misses, f"{len(misses)} of {len(TORCH_SHAPES) * 3} runs are not torch's bytes: {misses[:5]}"


def test_an_imported_configuration_is_the_heuristics_algorithm_byte_for_byte():
    """``encode_algo`` of what a pin reads back as is the pin (a pin is the encoded configuration), and runs to the same bytes."""
    for m, n, k in [(32, 1024, 1024), (320, 3072, 1024), (1, 1024, 7680), (64, 1024, 4096)]:
        for with_bias in (False, True):
            pin = torch_pin(m, n, k, with_bias)
            assert L.encode_algo(L.algo_info(pin.algo_hex)) == pin.algo_hex
            a, b, bias = inputs(m, n, k, with_bias, seed=m)
            assert same(run_allowing_bf16(a, b, bias, pin), run_allowing_bf16(a, b, bias, L.Pinned(L.encode_algo(L.algo_info(pin.algo_hex)), 0, pin.workspace_limit)))


def test_the_heuristic_pin_is_the_same_in_another_process():
    """A pin is the encoded configuration, so it does not carry bytes that change from process to process (measured on an RTX 5090 in what the
    heuristic itself returns for a split-K of id 21)."""
    shapes = [(1, 1024, 7680, False), (2, 1024, 7680, False), (32, 1024, 4096, True)]
    here = [torch_pin(*s).algo_hex for s in shapes]
    code = textwrap.dedent(
        f"""
        import json, torch
        from tilefish import _cublaslt as L
        torch.zeros(1, device="cuda")
        print(json.dumps([L.pin_algo(m, n, k, b, heuristic="torch", allow_bf16_reduction=True,
                          workspace_bytes=L.TORCH_LT_WORKSPACE_BYTES if b else L.DEFAULT_WORKSPACE_BYTES).algo_hex for m, n, k, b in {shapes!r}]))
        """
    )
    for _ in range(2):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert json.loads(out.stdout.strip().splitlines()[-1]) == here


def bf16_split(m, n, k, with_bias, scheme):
    """A split-K of the heuristic's algorithm for this problem that adds its partial sums in bf16 (``scheme``: 1 INPLACE, 4 OUTPUT_TYPE), or
    ``None`` when the library does not take one on this device."""
    base = L.algo_info(torch_pin(m, n, k, with_bias).algo_hex)
    for splitk in (2, 3, 4):
        algo = L.encode_algo({**base, "splitk_num": splitk, "reduction_scheme": scheme})
        try:
            L.check_algo(m, n, k, with_bias, algo, workspace_bytes=L.DEFAULT_WORKSPACE_BYTES, allow_bf16_reduction=True)
        except TilefishRuntimeError:
            continue
        return algo
    return None


@pytest.mark.parametrize("scheme", [1, 4])
def test_bf16_partial_sums_run_only_with_the_opt_in(scheme):
    m, n, k = 32, 1024, 4096
    algo = bf16_split(m, n, k, True, scheme)
    if algo is None:
        pytest.skip(f"cuBLASLt takes no split-K with reduction scheme {scheme} for {m}x{n}x{k} on this device")
    with pytest.raises(TilefishRuntimeError, match=f"reduction scheme {scheme} adds the split-K partial sums in bf16, which rounds more than once"):
        L.check_algo(m, n, k, True, algo, workspace_bytes=L.DEFAULT_WORKSPACE_BYTES)
    a, b, bias = inputs(m, n, k, True, seed=5)
    d = torch.empty((m, n), dtype=BF16, device=DEVICE)
    with pytest.raises(TilefishRuntimeError, match="adds the split-K partial sums in bf16"):
        L.matmul(a.data_ptr(), b.data_ptr(), d.data_ptr(), bias.data_ptr(), m, n, k, algo, workspace_bytes=L.DEFAULT_WORKSPACE_BYTES, stream=current_stream())
    pin = L.Pinned(algo, 0, L.DEFAULT_WORKSPACE_BYTES)
    first, second = run_allowing_bf16(a, b, bias, pin), run_allowing_bf16(a, b, bias, pin)
    torch.cuda.synchronize()
    assert same(first, second)  # deterministic
    exact = a.double() @ b.double().t() + bias.double()
    assert (first.double() - exact).abs().max() <= 2**-6 * exact.abs().max()  # a few roundings, not one: close, not exact
