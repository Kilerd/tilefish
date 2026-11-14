"""Pinned cuBLASLt GEMMs of f32 operands and an f32 result, TF32 off, on a real GPU.

The binding (:mod:`tilefish._cublaslt` with ``Form("f32", "f32", layout)``) and the library op (``@tilefish.library`` over f32 tensors):

* the ``"torch"`` posture with ``allow_f32_reduction=True`` (torch never restricts the reduction of an f32 GEMM) pins what torch runs with TF32 off
  (``torch.backends.cuda.matmul.allow_tf32 = False``, ``torch.set_float32_matmul_precision("highest")``): the result is ``torch.mm`` /
  ``torch.addmm`` in f32 byte for byte, for ``a @ b.t()`` (``TN``) and ``a @ b`` (``NN``), over the shapes of a transformer's projections and of a
  convolution as a GEMM at several ``M``, three operand sets each (asserted), with torch's default cuBLAS workspace and, in a process of its own,
  with ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (32 MiB);
* the test can tell: with ``allow_tf32 = True`` torch's f32 ``mm`` is not the pin's bytes, and neither is the pin of the other layout;
* the default posture is the f64 reference within one f32 accumulation;
* TF32 and the bf16x9 emulation never run: a pin chosen under ``CUBLAS_COMPUTE_32F_FAST_TF32`` is refused, and in a process with
  ``NVIDIA_TF32_OVERRIDE=1`` (or, with cuBLAS 13, ``CUBLAS_EMULATE_SINGLE_PRECISION=1``) every f32 pin is refused, a clean one and its own;
* split-K partial sums in the output run only with ``allow_f32_reduction``, and run to the same bytes twice;
* ``jit`` and ``replay`` agree byte for byte in both layouts with and without a bias, ``reference`` is close; an f32 ``AlgoTable`` imports the
  heuristic's algorithm; a launch plan of f32 GEMMs runs the same in ``jit`` and ``replay``.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

from gpu_util import differing, randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, TilefishBuildError, TilefishRuntimeError, f32, runtime  # noqa: E402
from tilefish import _cublaslt as L  # noqa: E402

F32 = torch.float32
DEVICE = "cuda"
MIB = 1 << 20


@pytest.fixture(autouse=True, scope="module")
def _cuda_is_up():
    torch.zeros(1, device=DEVICE)


@pytest.fixture(autouse=True)
def _tf32_off():
    """torch's f32 matmuls with TF32 off (its defaults, said explicitly), and the settings given back afterwards."""
    before = (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    yield
    torch.backends.cuda.matmul.allow_tf32 = before[0]
    torch.set_float32_matmul_precision(before[1])


def stream():
    return torch.cuda.current_stream().cuda_stream


def operands(m, n, k, layout, bias, seed=0, kind="normal"):
    """``a [m, k]``, ``b`` (``[n, k]`` for TN, ``[k, n]`` for NN) and an f32 bias; ``kind`` ``"wide"`` spreads the exponents (times
    2**-20 .. 2**20), ``"traps"`` makes exact powers of two with 3% of them +-2**20 (sums that cancel, so another order of the sum shows)."""
    a = randn((m, k), F32, seed=seed)
    b = randn((n, k) if layout == "TN" else (k, n), F32, seed=seed + 1, scale=1 / k**0.5)
    c = randn((n,), F32, seed=seed + 2) if bias else None
    if kind != "normal":
        g = torch.Generator().manual_seed(seed)
        out = []
        for x in (a, b):
            v = x.cpu()
            sign = torch.where(v < 0, -1.0, 1.0)
            if kind == "wide":
                v = v * torch.pow(2.0, torch.randint(-20, 21, v.shape, generator=g).float())
            else:
                v = sign * torch.pow(2.0, torch.randint(-6, 3, v.shape, generator=g).float())
                v = torch.where(torch.rand(v.shape, generator=g) < 0.03, sign * 2.0**20, v)
            out.append(v.to(DEVICE))
        a, b = out
    return a, b, c


def run(a, b, bias, algo_hex, form, *, workspace_bytes, allow=False):
    m, k = a.shape
    n = b.shape[0] if form.layout == "TN" else b.shape[1]
    d = torch.empty((m, n), dtype=F32, device=DEVICE)
    L.matmul(
        a.data_ptr(), b.data_ptr(), d.data_ptr(), None if bias is None else bias.data_ptr(), m, n, k, algo_hex,
        workspace_bytes=workspace_bytes, stream=stream(), form=form, allow_f32_reduction=allow,
    )  # fmt: skip
    return d


def torch_of(a, b, bias, layout):
    rhs = b.t() if layout == "TN" else b
    return torch.addmm(bias, a, rhs) if bias is not None else torch.mm(a, rhs)


# (N, K): projections of a transformer, and convolutions as GEMMs (K = 7 * C, a pointwise one); M from one row to many
TORCH_NK = [(1024, 1024), (3072, 1024), (1024, 3072), (4096, 1024), (1536, 7168), (768, 5376), (96, 672), (1024, 4096), (512, 1536)]
TORCH_M = (1, 32, 64, 100, 320, 1000, 2048)


def torch_workspace(bias, mm_workspace=L.DEFAULT_WORKSPACE_BYTES):
    """torch's workspace for the op: its cuBLASLt workspace for ``addmm`` (``CUBLASLT_WORKSPACE_SIZE``, 1 MiB), its cuBLAS one for ``mm``."""
    return L.TORCH_LT_WORKSPACE_BYTES if bias else mm_workspace


def torch_misses(layout, bias, ms=TORCH_M, nks=TORCH_NK, mm_workspace=L.DEFAULT_WORKSPACE_BYTES):
    """Pin each problem under the ``"torch"`` posture with ``allow_f32_reduction=True`` and torch's workspace, run three operand sets, and return
    ``(runs, misses)``: the runs whose bytes are not torch's."""
    form = L.Form("f32", "f32", layout)
    workspace = torch_workspace(bias, mm_workspace)
    misses, runs = [], 0
    for m in ms:
        for n, k in nks:
            pin = L.pin_algo(m, n, k, bias, heuristic="torch", allow_f32_reduction=True, form=form, workspace_bytes=workspace)
            for seed, kind in enumerate(("normal", "wide", "traps")):
                a, b, c = operands(m, n, k, layout, bias, seed=seed * 11 + m + n, kind=kind)
                ours = run(a, b, c, pin.algo_hex, form, workspace_bytes=workspace, allow=True)
                theirs = torch_of(a, b, c, layout)
                torch.cuda.synchronize()
                runs += 1
                if not same_bytes(ours, theirs):
                    misses.append((m, n, k, kind, differing(ours, theirs), L.algo_info(pin.algo_hex)))
    return runs, misses


@pytest.mark.parametrize("bias", [False, True], ids=["mm", "addmm"])
@pytest.mark.parametrize("layout", ["TN", "NN"])
def test_the_torch_posture_pins_what_torch_runs_in_f32_with_tf32_off(layout, bias):
    """``torch.mm`` (``torch.addmm`` with a 1-D bias) of f32 tensors, TF32 off, in both layouts is the pinned result byte for byte."""
    runs, misses = torch_misses(layout, bias)
    assert runs == len(TORCH_M) * len(TORCH_NK) * 3
    assert not misses, f"{len(misses)} of {runs} runs are not torch's bytes: {misses[:5]}"


WORKSPACE_CONFIG = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {tests!r})
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    import test_gpu_gemm_f32 as T
    out = {{}}
    for layout in ("TN", "NN"):
        for bias in (False, True):
            runs, misses = T.torch_misses(layout, bias, mm_workspace=32 << 20)
            out[f"{{layout}}:{{bias}}"] = [runs, [list(x[:5]) for x in misses]]
    print(json.dumps(out))
    """
)


def test_the_torch_posture_is_torchs_with_a_cublas_workspace_of_32_mib(capsys):
    """In a process with ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (torch's cuBLAS workspace is 32 MiB there) the ``"torch"`` posture with that
    workspace for ``mm`` (and torch's 1 MiB cuBLASLt workspace for ``addmm``) is torch's bytes too."""
    tests = os.path.dirname(os.path.abspath(__file__))
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    out = subprocess.run([sys.executable, "-c", WORKSPACE_CONFIG.format(tests=tests)], capture_output=True, text=True, env=env, check=True)
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert all(runs == len(TORCH_M) * len(TORCH_NK) * 3 for runs, _ in got.values())
    assert all(not misses for _, misses in got.values()), {k: v[1][:3] for k, v in got.items() if v[1]}
    with capsys.disabled():
        print(f"\n[f32 torch posture] CUBLAS_WORKSPACE_CONFIG=:4096:8: {sum(r for r, _ in got.values())} runs, 0 misses")


def test_the_test_can_tell_tf32_and_another_layout_from_the_pin():
    """TRAP: with ``allow_tf32 = True`` torch's f32 ``mm`` rounds its inputs to TF32 on a GPU that has it, and is not the pin's bytes; the pin of
    the other layout (b read as ``[K, N]`` where it is ``[N, K]``) is not torch's ``a @ b.t()`` either."""
    form = L.Form("f32", "f32", "TN")
    tf32_differs = 0
    for m, n, k in [(32, 1024, 1024), (320, 3072, 1024), (2048, 1024, 4096)]:
        pin = L.pin_algo(m, n, k, False, heuristic="torch", allow_f32_reduction=True, form=form)
        a, b, _ = operands(m, n, k, "TN", False, seed=m)
        ours = run(a, b, None, pin.algo_hex, form, workspace_bytes=L.DEFAULT_WORKSPACE_BYTES, allow=True)
        assert same_bytes(ours, torch.mm(a, b.t()))
        torch.backends.cuda.matmul.allow_tf32 = True
        tf32 = torch.mm(a, b.t())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.synchronize()
        tf32_differs += differing(ours, tf32) > 0
        # TF32 keeps 10 bits of each input's mantissa: far from the f32 result
        assert (tf32 - ours).abs().max().item() > 1e-5 * ours.abs().max().item()
    assert tf32_differs == 3
    # the wrong layout: a square b read as [K, N] computes a @ b, not a @ b.t()
    pin = L.pin_algo(64, 1024, 1024, False, heuristic="torch", allow_f32_reduction=True, form=L.Form("f32", "f32", "NN"))
    a, b, _ = operands(64, 1024, 1024, "TN", False, seed=5)
    wrong = run(a, b, None, pin.algo_hex, L.Form("f32", "f32", "NN"), workspace_bytes=L.DEFAULT_WORKSPACE_BYTES, allow=True)
    assert differing(wrong, torch.mm(a, b.t())) > 1000 and same_bytes(wrong, torch.mm(a, b))


@pytest.mark.parametrize("layout", ["TN", "NN"])
@pytest.mark.parametrize("bias", [False, True])
def test_the_default_posture_is_the_f64_reference_within_one_f32_accumulation(layout, bias):
    form = L.Form("f32", "f32", layout)
    for m, n, k in [(32, 1024, 1024), (320, 3072, 1024), (1, 1536, 7168), (2048, 96, 672), (64, 1000, 1001)]:
        pin = L.pin_algo(m, n, k, bias, form=form)
        a, b, c = operands(m, n, k, layout, bias, seed=m + k)
        got = run(a, b, c, pin.algo_hex, form, workspace_bytes=pin.workspace_limit).double().cpu()
        a64, b64 = a.double().cpu(), b.double().cpu()
        rhs = b64.t() if layout == "TN" else b64
        exact = a64 @ rhs + (c.double().cpu() if bias else 0)
        absolute = a64.abs() @ rhs.abs() + (c.double().cpu().abs() if bias else 0)
        gamma = (k + 1) * 2.0**-24 / (1 - (k + 1) * 2.0**-24)
        excess = ((got - exact).abs() - gamma * absolute - (2.0**-24 * exact.abs() + 2.0**-150) * 1.0001).max().item()
        assert excess <= 0, (m, n, k, layout, bias, excess)


# -- TF32 and the emulation never run -------------------------------------------------------------------------------------------------------


def pin_under(compute, m, n, k, layout):
    """The pin the heuristic gives under another compute type (``CUBLAS_COMPUTE_32F_FAST_TF32`` is 77): what a pin made with TF32 on is."""
    saved = L.CUBLAS_COMPUTE_32F
    L.CUBLAS_COMPUTE_32F = compute
    L._validated.clear()
    try:
        return L.pin_algo(m, n, k, False, heuristic="torch", allow_f32_reduction=True, form=L.Form("f32", "f32", layout)).algo_hex
    finally:
        L.CUBLAS_COMPUTE_32F = saved
        L._validated.clear()


def test_a_pin_chosen_with_tf32_on_is_refused():
    refused = 0
    for m, n, k in [(32, 1024, 1024), (320, 1024, 3072), (2048, 96, 672)]:
        for layout in ("TN", "NN"):
            algo = pin_under(77, m, n, k, layout)
            with pytest.raises(TilefishRuntimeError, match="is not one cuBLASLt produces for f32 with f32 accumulation") as e:
                L.check_algo(m, n, k, False, algo, workspace_bytes=L.DEFAULT_WORKSPACE_BYTES, form=L.Form("f32", "f32", layout), allow_f32_reduction=True)
            # an id of the TF32 family is not encodable under CUBLAS_COMPUTE_32F; another one is encoded with another compute type
            assert "is unknown" in str(e.value) or "its bytes are not those cuBLASLt encodes" in str(e.value) or "read the f32 operands as tf32" in str(e.value)
            refused += 1
    assert refused == 6


ENVIRONMENT = textwrap.dedent(
    """
    import json, sys
    import torch
    from tilefish import _cublaslt as L
    clean = json.loads({clean!r})
    out = {{"version": L.version()}}
    for key, (m, n, k, layout, algo) in clean.items():
        form = L.Form("f32", "f32", layout)
        try:
            L.check_algo(m, n, k, False, algo, workspace_bytes=L.DEFAULT_WORKSPACE_BYTES, form=form, allow_f32_reduction=True)
            out[key] = "accepted"
        except Exception as e:
            out[key] = str(e)
        try:
            own = L.pin_algo(m, n, k, False, heuristic="torch", allow_f32_reduction=True, form=form).algo_hex
            L.check_algo(m, n, k, False, own, workspace_bytes=L.DEFAULT_WORKSPACE_BYTES, form=form, allow_f32_reduction=True)
            out[key + ":own"] = "accepted"
        except Exception as e:
            out[key + ":own"] = str(e)
    print(json.dumps(out))
    """
)


@pytest.mark.parametrize("variable", ["NVIDIA_TF32_OVERRIDE", "CUBLAS_EMULATE_SINGLE_PRECISION"])
def test_a_process_whose_cublaslt_runs_f32_as_tf32_or_an_emulation_refuses_every_f32_pin(variable):
    """With ``NVIDIA_TF32_OVERRIDE=1`` cuBLASLt encodes and runs ``CUBLAS_COMPUTE_32F`` as ``_FAST_TF32``, and with
    ``CUBLAS_EMULATE_SINGLE_PRECISION=1`` (cuBLAS 13) as ``_EMULATED_16BFX9``: there a pin of a clean process is refused (its bytes), and so is the
    process's own (its compute type). cuBLAS 12.8 has no emulation: the variable changes nothing there."""
    clean = {}
    for m, n, k in [(1, 1024, 1024), (32, 1024, 1024), (320, 1024, 3072)]:
        for layout in ("TN", "NN"):
            algo = L.pin_algo(m, n, k, False, heuristic="torch", allow_f32_reduction=True, form=L.Form("f32", "f32", layout)).algo_hex
            clean[f"{layout}:{m}x{n}x{k}"] = (m, n, k, layout, algo)
    env = {**os.environ, variable: "1"}
    out = subprocess.run([sys.executable, "-c", ENVIRONMENT.format(clean=json.dumps(clean))], capture_output=True, text=True, env=env, check=True)
    got = json.loads(out.stdout.strip().splitlines()[-1])
    if variable == "CUBLAS_EMULATE_SINGLE_PRECISION" and got["version"] < 120900:
        assert all(v == "accepted" for key, v in got.items() if key != "version"), got
        return
    promoted = "CUBLAS_COMPUTE_32F_FAST_TF32 (77)" if variable == "NVIDIA_TF32_OVERRIDE" else "CUBLAS_COMPUTE_32F_EMULATED_16BFX9 (78)"
    for key in clean:
        assert got[key] != "accepted" and got[key + ":own"] != "accepted", (key, got[key], got[key + ":own"])
        assert f"cuBLASLt encodes CUBLAS_COMPUTE_32F as {promoted} in this process" in got[key + ":own"], got[key + ":own"]


# -- split-K in the output ----------------------------------------------------------------------------------------------------------------


def test_split_k_partial_sums_in_the_output_run_only_with_the_opt_in_and_run_to_the_same_bytes():
    found = None
    for m, n, k in [(32, 1024, 4096), (100, 1536, 7168), (64, 3072, 1024), (32, 1024, 1024), (320, 1024, 3072), (1000, 512, 1536)]:
        for layout in ("TN", "NN"):
            pin = L.pin_algo(m, n, k, False, heuristic="torch", allow_f32_reduction=True, form=L.Form("f32", "f32", layout))
            if L.algo_info(pin.algo_hex)["reduction_scheme"] in (1, 4):
                found = (m, n, k, layout, pin)
                break
        if found:
            break
    if found is None:
        pytest.skip("the torch posture chooses no split-K in the output for these problems on this device")
    m, n, k, layout, pin = found
    form = L.Form("f32", "f32", layout)
    scheme = L.algo_info(pin.algo_hex)["reduction_scheme"]
    with pytest.raises(TilefishRuntimeError, match=f"reduction scheme {scheme} adds the split-K partial sums in the output \\(f32\\).*allow_f32_reduction=True"):
        L.check_algo(m, n, k, False, pin.algo_hex, workspace_bytes=L.DEFAULT_WORKSPACE_BYTES, form=form)
    a, b, _ = operands(m, n, k, layout, False, seed=9)
    first = run(a, b, None, pin.algo_hex, form, workspace_bytes=L.DEFAULT_WORKSPACE_BYTES, allow=True)
    second = run(a, b, None, pin.algo_hex, form, workspace_bytes=L.DEFAULT_WORKSPACE_BYTES, allow=True)
    torch.cuda.synchronize()
    assert same_bytes(first, second) and same_bytes(first, torch_of(a, b, None, layout))


# -- the library op: jit, replay, reference, plans, imported tables -------------------------------------------------------------------------

M = Dim("M", buckets=[32, 320])


def declarations(**kw):
    """f32 GEMMs of each kind, over the buckets of ``M``: TN without a bias, NN with one, TN with one."""

    @tilefish.library("f32_tn", acc=f32, impl="cublaslt.gemm", **kw)
    def tn(a: Tensor[M, 1024, f32], b: Tensor[3072, 1024, f32]) -> Tensor[M, 3072, f32]: ...

    @tilefish.library("f32_nn_bias", acc=f32, impl="cublaslt.gemm_bias", layout="NN", **kw)
    def nn(a: Tensor[M, 7168, f32], b: Tensor[7168, 1536, f32], bias: Tensor[1536, f32]) -> Tensor[M, 1536, f32]: ...

    @tilefish.library("f32_tn_bias", acc=f32, impl="cublaslt.gemm_bias", **kw)
    def tn_bias(a: Tensor[M, 3072, f32], b: Tensor[1024, 3072, f32], bias: Tensor[1024, f32]) -> Tensor[M, 1024, f32]: ...

    return {"tn": (tn, 3072, 1024, "TN", False), "nn": (nn, 1536, 7168, "NN", True), "tn_bias": (tn_bias, 1024, 3072, "TN", True)}


def test_jit_and_replay_agree_for_f32_in_both_layouts_and_reference_is_close(tmp_path):
    ops = declarations()
    manifest = runtime.export([op for op, *_ in ops.values()], tmp_path)
    entries = {e["stem"]: e for e in manifest.library_entries}
    assert entries["f32_nn_bias.M32"]["dtypes"] == {"a": "f32", "b": "f32", "bias": "f32", "out": "f32"}
    assert entries["f32_nn_bias.M32"]["transa"] == "N" and entries["f32_nn_bias.M32"]["lda"] == 1536
    for op, n, k, layout, bias in ops.values():
        for m in (32, 320):
            a, b, c = operands(m, n, k, layout, bias, seed=m + n)
            args = (a, b, c) if bias else (a, b)
            jit = tilefish.run(op, *args)
            replay = tilefish.run(op, *args, mode="replay", manifest=tmp_path, pin=manifest.digest)
            ref = tilefish.run(op, *args, mode="reference")
            assert jit.dtype == F32 and same_bytes(jit, replay), (op.name, m)
            assert (jit.double() - ref.double()).abs().max() <= 1e-4 * ref.double().abs().max() + 1e-5, (op.name, m)


def test_the_torch_posture_of_an_f32_declaration_is_torchs_in_both_layouts(tmp_path):
    ops = declarations(heuristic="torch", allow_f32_reduction=True)
    manifest = runtime.export([op for op, *_ in ops.values()], tmp_path)
    assert all(e.get("allow_f32_reduction") is True and e.get("heuristic") == "torch" for e in manifest.library_entries)
    assert {e["stem"]: e["workspace_bytes"] for e in manifest.library_entries}["f32_nn_bias.M32"] == L.TORCH_LT_WORKSPACE_BYTES
    for op, n, k, layout, bias in ops.values():
        for m in (32, 320):
            a, b, c = operands(m, n, k, layout, bias, seed=m)
            args = (a, b, c) if bias else (a, b)
            replay = tilefish.run(op, *args, mode="replay", manifest=tmp_path, pin=manifest.digest)
            assert same_bytes(replay, torch_of(a, b, c, layout)), (op.name, m)


def test_an_imported_f32_table_runs_the_heuristics_algorithm(tmp_path):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability()
    form = L.Form("f32", "f32", "NN")
    rows = {(m, 1536, 7168, True, "NN"): L.algo_info(L.pin_algo(m, 1536, 7168, True, form=form).algo_hex) for m in (32, 320)}
    table = tilefish.AlgoTable(f"sm_{major * 10 + minor}", props.multi_processor_count, L.version(), rows)
    plain = declarations()["nn"][0]

    @tilefish.library("f32_nn_bias", acc=f32, impl="cublaslt.gemm_bias", layout="NN", algos=table)
    def imported(a: Tensor[M, 7168, f32], b: Tensor[7168, 1536, f32], bias: Tensor[1536, f32]) -> Tensor[M, 1536, f32]: ...

    m_imported = runtime.export([imported], tmp_path / "imported")
    m_plain = runtime.export([plain], tmp_path / "plain")
    for m in (32, 320):
        stem = f"f32_nn_bias.M{m}"
        assert m_imported.library_entry(stem, table.arch)["algo"] == m_plain.library_entry(stem, table.arch)["algo"]
        assert m_imported.library_entry(stem, table.arch)["algo_source"] == "imported"
        a, b, c = operands(m, 1536, 7168, "NN", True, seed=m)
        assert same_bytes(
            tilefish.run(imported, a, b, c, mode="replay", manifest=tmp_path / "imported", pin=m_imported.digest),
            tilefish.run(plain, a, b, c, mode="replay", manifest=tmp_path / "plain", pin=m_plain.digest),
        )
    # an algorithm encodes its types: the f32 pin is refused for bf16 operands (another encoding, or an id with no bf16 kernels: 76 on sm_103)
    f32_algo = m_plain.library_entry("f32_nn_bias.M32", table.arch)["algo"]
    with pytest.raises(TilefishRuntimeError, match="is not one cuBLASLt produces for bf16 with f32 accumulation"):
        L.check_algo(32, 1536, 7168, True, f32_algo, workspace_bytes=L.DEFAULT_WORKSPACE_BYTES, form=L.Form("bf16", "bf16", "NN"))


def test_a_plan_of_f32_gemms_runs_the_same_in_jit_replay_and_reference(tmp_path):
    """A graph of an NN GEMM with a bias into a TN GEMM with a bias, all f32: jit and replay are byte for byte the same and the GEMMs run in the
    plan as they run alone; the reference (torch in f32) is close."""
    nn = declarations()["nn"][0]

    @tilefish.library("f32_proj", acc=f32, impl="cublaslt.gemm_bias", workspace_bytes=64 << 20)
    def proj(a: Tensor[M, 1536, f32], b: Tensor[1024, 1536, f32], bias: Tensor[1024, f32]) -> Tensor[M, 1024, f32]: ...

    @tilefish.graph(inputs=("x",), weights=("w1", "b1", "w2", "b2"))
    def block(x: Tensor[M, 7168, f32], w1: Tensor[7168, 1536, f32], b1: Tensor[1536, f32], w2: Tensor[1024, 1536, f32], b2: Tensor[1024, f32]):
        return proj(nn(x, w1, b1), w2, b2)

    plan = runtime.export(block, tmp_path)
    for m in (32, 320):
        x, w1, b1 = operands(m, 1536, 7168, "NN", True, seed=m)
        _, w2, b2 = operands(m, 1024, 1536, "TN", True, seed=m + 5)
        args = (x, w1, b1, w2, b2)
        jit = tilefish.run(block, *args)
        replay = tilefish.run(block, *args, mode="replay", plan=tmp_path, pin=plan.digest)
        ref = tilefish.run(block, *args, mode="reference")
        assert jit.dtype == F32 and same_bytes(jit, replay), m
        direct = tilefish.run(proj, tilefish.run(nn, x, w1, b1), w2, b2)
        assert same_bytes(jit, direct), m
        assert (jit.double() - ref.double()).abs().max() <= 1e-4 * ref.double().abs().max() + 1e-5, m


def test_an_f32_declaration_is_refused_at_the_build_where_cublaslt_runs_f32_as_tf32(tmp_path):
    """The build pins and then checks: in a process with ``NVIDIA_TF32_OVERRIDE=1`` an export of an f32 GEMM is refused, never written."""
    tests = os.path.dirname(os.path.abspath(__file__))
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {tests!r})
        import test_gpu_gemm_f32 as T
        from tilefish import runtime
        try:
            runtime.export([T.declarations()["tn"][0]], {str(tmp_path)!r})
            print("written")
        except Exception as e:
            print(type(e).__name__, str(e).replace(chr(10), " "))
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env={**os.environ, "NVIDIA_TF32_OVERRIDE": "1"}, check=True)
    last = out.stdout.strip().splitlines()[-1]
    assert last.startswith(TilefishBuildError.__name__) and "cuBLASLt encodes CUBLAS_COMPUTE_32F as CUBLAS_COMPUTE_32F_FAST_TF32 (77)" in last, last
