"""Pinned cuBLASLt GEMMs of the layouts ``TT`` and ``NT`` (``a`` stored ``[K, M]``), for bf16, f16 and f32 operands, on a real GPU.

* the ``"torch"`` posture with the opt-in of the operands' type (torch's default for bf16 and f16; f32 with TF32 off) pins what torch runs for
  ``torch.mm(a.t(), b.t())`` (``TT``) and ``torch.mm(a.t(), b)`` (``NT``), and ``torch.addmm`` with a 1-D bias, byte for byte, over the shapes of a
  transformer's projections, ``K = 8`` and several ``M`` (asserted per operand type; the f32 cases are what torch's ``cublasSgemm`` runs);
* the result of the default posture is the f64 reference within one f32 accumulation and one rounding;
* ``jit`` and ``replay`` agree byte for byte for every operand type, with and without a bias; a launch plan of them (and its CUDA graph) is the same;
* a configuration imported with ``AlgoTable`` (a key of the layout says it) is the heuristic's algorithm.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

from gpu_util import differing, randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, TilefishRuntimeError, bf16, f16, f32, runtime  # noqa: E402
from tilefish import _cublaslt as L  # noqa: E402

DEVICE = "cuda"
TORCH = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
TYPES = {"bf16": bf16, "f16": f16, "f32": f32}


@pytest.fixture(autouse=True, scope="module")
def _cuda_is_up():
    torch.zeros(1, device=DEVICE)


@pytest.fixture(autouse=True)
def _tf32_off():
    before = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = before


def operands(m, n, k, layout, bias, dtype, seed=0):
    """``a`` (``[k, m]``: the layouts here store it transposed), ``b`` (``[n, k]`` for TT, ``[k, n]`` for NT) and a bias of the operands' type."""
    t = TORCH[dtype]
    a = randn((k, m) if layout[1] == "T" else (m, k), t, seed=seed)
    b = randn((n, k) if layout[0] == "T" else (k, n), t, seed=seed + 1, scale=1 / k**0.5)
    c = randn((n,), t, seed=seed + 2) if bias else None
    return a, b, c


def run(a, b, bias, pin, form, allow):
    m = a.shape[1] if form.layout[1] == "T" else a.shape[0]
    k = a.shape[0] if form.layout[1] == "T" else a.shape[1]
    n = b.shape[0] if form.layout[0] == "T" else b.shape[1]
    d = torch.empty((m, n), dtype=TORCH[form.out], device=DEVICE)
    L.matmul(
        a.data_ptr(), b.data_ptr(), d.data_ptr(), None if bias is None else bias.data_ptr(), m, n, k, pin.algo_hex,
        workspace_bytes=pin.workspace_limit, stream=torch.cuda.current_stream().cuda_stream, form=form,
        **{form.opt_in: allow},
    )  # fmt: skip
    return d


def torch_of(a, b, bias, layout):
    lhs = a.t() if layout[1] == "T" else a
    rhs = b.t() if layout[0] == "T" else b
    return torch.addmm(bias, lhs, rhs) if bias is not None else torch.mm(lhs, rhs)


# (N, K): projections, a pointwise convolution as a GEMM, and K = 8; M from two rows to many. (At M = 1, a [K, 1] is also a contiguous [1, K],
# which torch multiplies as the layout TN or NN: there torch's bytes are those of that layout, test_gpu_gemm_f16.py and test_gpu_gemm_f32.py. Measured:
# torch.addmm of a bf16 a [1024, 1].t() is not the TT pin's bytes at 2 of 6 shapes.)
TORCH_NK = [(1024, 1024), (3072, 1024), (1024, 3072), (96, 672), (1024, 8), (4096, 8)]
TORCH_M = (2, 32, 100, 320, 646, 2048)


def torch_pin(m, n, k, bias, form):
    """The ``"torch"`` posture with the opt-in of the operands' type and torch's workspace for the op (``addmm``: its cuBLASLt workspace)."""
    return L.pin_algo(
        m, n, k, bias, heuristic="torch", form=form, workspace_bytes=L.TORCH_LT_WORKSPACE_BYTES if bias else L.DEFAULT_WORKSPACE_BYTES,
        **{form.opt_in: True},
    )  # fmt: skip


@pytest.mark.parametrize("bias", [False, True], ids=["mm", "addmm"])
@pytest.mark.parametrize("layout", ["TT", "NT"])
@pytest.mark.parametrize("dtype", ["bf16", "f16", "f32"])
def test_the_torch_posture_pins_what_torch_runs_for_a_transposed_a(dtype, layout, bias):
    """``torch.mm(a.t(), ...)`` (``torch.addmm``) of a contiguous ``a [k, m]`` is the pinned result of the layout byte for byte."""
    form = L.Form(dtype, dtype, layout)
    misses, runs, refused = [], 0, []
    for m in TORCH_M:
        for n, k in TORCH_NK:
            pin = torch_pin(m, n, k, bias, form)
            if dtype == "f32" and m == 2:
                # measured with cuBLASLt 13.1.0 on an RTX 5090 (sm_120): at M = 2 the heuristic answers algorithm 10 with a split-K count of 0 for
                # some f32 problems of a stored [K, M], which the vetting of both hosts refuses ("split-K count 0 is below 1"); elsewhere it is run
                try:
                    L.check_algo(m, n, k, bias, pin.algo_hex, workspace_bytes=pin.workspace_limit, form=form, allow_f32_reduction=True)
                except TilefishRuntimeError as e:
                    assert "split-K count 0 is below 1" in str(e) and L.algo_info(pin.algo_hex)["id"] == 10, e
                    refused.append((m, n, k))
                    runs += 2
                    continue
            for seed in range(2):
                a, b, c = operands(m, n, k, layout, bias, dtype, seed=seed * 11 + m + n)
                ours = run(a, b, c, pin, form, allow=True)
                theirs = torch_of(a, b, c, layout)
                torch.cuda.synchronize()
                runs += 1
                if not same_bytes(ours, theirs):
                    misses.append((m, n, k, seed, differing(ours, theirs), L.algo_info(pin.algo_hex)))
    assert runs == len(TORCH_M) * len(TORCH_NK) * 2
    assert not misses, f"{len(misses)} of {runs} runs are not torch's bytes: {misses[:5]}"
    if refused:
        print(f"\n{dtype} {layout} bias={bias}: the heuristic's answer is refused at {refused}")


@pytest.mark.parametrize("layout", ["TT", "NT"])
@pytest.mark.parametrize("dtype,out", [("bf16", "bf16"), ("f16", "f32"), ("f32", "f32")])
@pytest.mark.parametrize("bias", [False, True])
def test_the_default_posture_is_the_f64_reference_within_one_accumulation_and_one_rounding(layout, dtype, out, bias):
    form = L.Form(dtype, out, layout)
    unit = {"bf16": 2.0**-8, "f16": 2.0**-11, "f32": 2.0**-24}[out]
    tiny = {"bf16": 2.0**-133, "f16": 2.0**-25, "f32": 2.0**-150}[out]
    for m, n, k in [(32, 1024, 1024), (320, 3072, 1024), (1, 1536, 768), (2048, 96, 672), (646, 1024, 8), (64, 1000, 1001)]:
        pin = L.pin_algo(m, n, k, bias, form=form)
        a, b, _ = operands(m, n, k, layout, False, dtype, seed=m + k)
        c = randn((n,), TORCH[out], seed=m) if bias else None
        got = run(a, b, c, pin, form, allow=False).double().cpu()
        a64 = a.double().cpu()
        a64 = a64.t() if layout[1] == "T" else a64
        b64 = b.double().cpu()
        rhs = b64.t() if layout[0] == "T" else b64
        exact = a64 @ rhs + (c.double().cpu() if bias else 0)
        absolute = a64.abs() @ rhs.abs() + (c.double().cpu().abs() if bias else 0)
        gamma = (k + 1) * 2.0**-24 / (1 - (k + 1) * 2.0**-24)
        excess = ((got - exact).abs() - gamma * absolute - (unit * exact.abs() + tiny) * 1.0001).max().item()
        assert excess <= 0, (m, n, k, layout, dtype, out, bias, excess)


# -- the library op: jit, replay, plans, imported tables -------------------------------------------------------------------------------------

M = Dim("M", buckets=[32, 320])


def declarations():
    """A GEMM of each operand type with ``a [K, M]``: bf16 in TT under the torch posture, f16 in NT with a bias, f32 in TT with K = 8 and f32 in NT
    with a bias (32 MiB of workspace)."""

    @tilefish.library("bf16_tt", acc=f32, impl="cublaslt.gemm", layout="TT", heuristic="torch", allow_bf16_reduction=True)
    def bf16_tt(a: Tensor[1024, M, bf16], b: Tensor[3072, 1024, bf16]) -> Tensor[M, 3072, bf16]: ...

    @tilefish.library("f16_nt_bias", acc=f32, impl="cublaslt.gemm_bias", layout="NT")
    def f16_nt(a: Tensor[3072, M, f16], b: Tensor[3072, 1536, f16], bias: Tensor[1536, f16]) -> Tensor[M, 1536, f16]: ...

    @tilefish.library("f32_tt_k8", acc=f32, impl="cublaslt.gemm", layout="TT")
    def f32_tt(a: Tensor[8, M, f32], b: Tensor[4096, 8, f32]) -> Tensor[M, 4096, f32]: ...

    @tilefish.library("f32_nt_bias", acc=f32, impl="cublaslt.gemm_bias", layout="NT", workspace_bytes=32 << 20)
    def f32_nt(a: Tensor[672, M, f32], b: Tensor[672, 96, f32], bias: Tensor[96, f32]) -> Tensor[M, 96, f32]: ...

    return {
        "bf16_tt": (bf16_tt, 3072, 1024, "TT", False, "bf16"),
        "f16_nt": (f16_nt, 1536, 3072, "NT", True, "f16"),
        "f32_tt": (f32_tt, 4096, 8, "TT", False, "f32"),
        "f32_nt": (f32_nt, 96, 672, "NT", True, "f32"),
    }


def test_jit_and_replay_agree_for_every_operand_type_with_a_transposed_a(tmp_path):
    ops = declarations()
    manifest = runtime.export([op for op, *_ in ops.values()], tmp_path)
    entries = {e["stem"]: e for e in manifest.library_entries}
    assert (entries["f16_nt_bias.M32"]["transa"], entries["f16_nt_bias.M32"]["transb"], entries["f16_nt_bias.M32"]["ldb"]) == ("N", "T", 32)
    assert (entries["f32_tt_k8.M320"]["transa"], entries["f32_tt_k8.M320"]["transb"], entries["f32_tt_k8.M320"]["ldb"]) == ("T", "T", 320)
    for op, n, k, layout, bias, dtype in ops.values():
        for m in (32, 320):
            a, b, c = operands(m, n, k, layout, bias, dtype, seed=m + n)
            args = (a, b, c) if bias else (a, b)
            jit = tilefish.run(op, *args)
            replay = tilefish.run(op, *args, mode="replay", manifest=tmp_path, pin=manifest.digest)
            ref = tilefish.run(op, *args, mode="reference")
            assert jit.shape == (m, n) and same_bytes(jit, replay), (op.name, m)
            assert (jit.double() - ref.double()).abs().max() <= 2**-7 * ref.double().abs().max() + 1e-3, (op.name, m)
            if op.name == "bf16_tt":  # the torch posture: torch's bytes
                assert same_bytes(replay, torch_of(a, b, None, layout)), m


def test_a_plan_of_transposed_gemms_runs_the_same_in_jit_replay_capture_and_alone(tmp_path):
    """``y = (x^T @ w1)^T``-like chains: an NT GEMM whose result is the stored ``[K, M]`` of the next TT GEMM (through a transposed view is not a
    plan's business: the second GEMM takes ``h [M, N]`` as ``a [K', M']`` of its own shape)."""

    @tilefish.library("nt_proj", acc=f32, impl="cublaslt.gemm_bias", layout="NT")
    def nt_proj(a: Tensor[1024, M, f16], b: Tensor[1024, 1024, f16], bias: Tensor[1024, f16]) -> Tensor[M, 1024, f16]: ...

    @tilefish.library("tt_mix", acc=f32, impl="cublaslt.gemm", layout="TT")
    def tt_mix(a: Tensor[M, 1024, f16], b: Tensor[512, M, f16]) -> Tensor[1024, 512, f16]: ...

    @tilefish.graph(inputs=("x",), weights=("w1", "b1", "w2"))
    def block(x: Tensor[1024, M, f16], w1: Tensor[1024, 1024, f16], b1: Tensor[1024, f16], w2: Tensor[512, M, f16]):
        return tt_mix(nt_proj(x, w1, b1), w2)

    plan = runtime.export(block, tmp_path)
    for m in (32, 320):
        x = randn((1024, m), torch.float16, seed=m)
        w1 = randn((1024, 1024), torch.float16, seed=m + 1, scale=1 / 32)
        b1 = randn((1024,), torch.float16, seed=m + 2)
        w2 = randn((512, m), torch.float16, seed=m + 3, scale=1 / m**0.5)
        args = (x, w1, b1, w2)
        jit = tilefish.run(block, *args)
        replay = tilefish.run(block, *args, mode="replay", plan=tmp_path, pin=plan.digest)
        direct = tilefish.run(tt_mix, tilefish.run(nt_proj, x, w1, b1), w2)
        assert same_bytes(jit, replay) and same_bytes(jit, direct), m
        out = torch.empty_like(jit)
        runtime.prepare(block, mode="replay", plan=tmp_path, pin=plan.digest)
        graph = tilefish.capture(block, *args, mode="replay", plan=tmp_path, pin=plan.digest, outputs=out)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert same_bytes(out, replay), m


def test_an_imported_table_of_the_layout_tt_runs_the_heuristics_algorithm(tmp_path):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability()
    form = L.Form("f32", "f32", "TT")
    rows = {(m, 4096, 8, False, "TT"): L.algo_info(L.pin_algo(m, 4096, 8, False, form=form).algo_hex) for m in (32, 320)}
    table = tilefish.AlgoTable(f"sm_{major * 10 + minor}", props.multi_processor_count, L.version(), rows)
    plain = declarations()["f32_tt"][0]

    @tilefish.library("f32_tt_k8", acc=f32, impl="cublaslt.gemm", layout="TT", algos=table)
    def imported(a: Tensor[8, M, f32], b: Tensor[4096, 8, f32]) -> Tensor[M, 4096, f32]: ...

    m_imported = runtime.export([imported], tmp_path / "imported")
    m_plain = runtime.export([plain], tmp_path / "plain")
    for m in (32, 320):
        stem = f"f32_tt_k8.M{m}"
        assert m_imported.library_entry(stem, table.arch)["algo"] == m_plain.library_entry(stem, table.arch)["algo"]
        assert m_imported.library_entry(stem, table.arch)["algo_source"] == "imported"
        a, b, _ = operands(m, 4096, 8, "TT", False, "f32", seed=m)
        assert same_bytes(
            tilefish.run(imported, a, b, mode="replay", manifest=tmp_path / "imported", pin=m_imported.digest),
            tilefish.run(plain, a, b, mode="replay", manifest=tmp_path / "plain", pin=m_plain.digest),
        )
