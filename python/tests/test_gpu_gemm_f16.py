"""Pinned cuBLASLt GEMMs of f16 operands, of an f32 result and of the layout ``NN``, on a real GPU.

The binding (:mod:`tilefish._cublaslt` with a :class:`~tilefish._cublaslt.Form`) and the library op (``@tilefish.library(..., layout=...)``):

* the ``"torch"`` posture with ``allow_f16_reduction=True`` (torch's default ``allow_fp16_reduced_precision_reduction``) pins what torch runs: the
  result is ``torch.mm`` / ``torch.addmm`` in f16 byte for byte, for ``a @ b.t()`` (``TN``) and ``a @ b`` (``NN``), over the shapes of a transformer's
  projections and of a convolution as a GEMM at several ``M``, and three operand sets each (asserted);
* the result of the default posture is the f64 reference within one f32 accumulation and one rounding to f16 or f32;
* ``jit`` and ``replay`` agree byte for byte, for f16 and f32 results, a bias, both layouts and a workspace of 64 MiB; a launch plan of them is the
  same; a configuration imported with ``AlgoTable`` (a key of the layout ``NN`` says it) is the heuristic's algorithm, encoded with f16 types;
* a split-K that adds its partial sums in f16 runs only with ``allow_f16_reduction``;
* a pin of one form is refused for another (an algorithm encodes its types).
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, TilefishRuntimeError, f16, f32, runtime  # noqa: E402
from tilefish import _cublaslt as L  # noqa: E402

F16 = torch.float16
F32 = torch.float32
DEVICE = "cuda"


@pytest.fixture(autouse=True, scope="module")
def _cuda_is_up():
    torch.zeros(1, device=DEVICE)


def stream():
    return torch.cuda.current_stream().cuda_stream


def operands(m, n, k, layout, bias, out=F16, seed=0, scale=None):
    """``a [m, k]``, ``b`` (``[n, k]`` for TN, ``[k, n]`` for NN) and a bias of the result's type, f16 values of a sensible magnitude."""
    a = randn((m, k), F16, seed=seed)
    shape = (n, k) if layout == "TN" else (k, n)
    b = randn(shape, F16, seed=seed + 1, scale=scale if scale is not None else 1 / k**0.5)
    c = randn((n,), out, seed=seed + 2) if bias else None
    return a, b, c


def run(a, b, bias, pin, form, *, allow=False):
    m, k = a.shape
    n = b.shape[0] if form.layout == "TN" else b.shape[1]
    d = torch.empty((m, n), dtype=F32 if form.out == "f32" else F16, device=DEVICE)
    L.matmul(
        a.data_ptr(), b.data_ptr(), d.data_ptr(), None if bias is None else bias.data_ptr(), m, n, k, pin.algo_hex,
        workspace_bytes=pin.workspace_limit, stream=stream(), form=form, allow_f16_reduction=allow,
    )  # fmt: skip
    return d


def torch_of(a, b, bias, layout):
    rhs = b.t() if layout == "TN" else b
    return torch.addmm(bias, a, rhs) if bias is not None else torch.mm(a, rhs)


def special(x, kind, seed):
    """``x`` made into f16 values of many exponents (``"wide"``: times 2**-10 .. 2**5) or into exact powers of two with 3% of them +-2**10
    (``"traps"``: sums that cancel, so another order of the sum shows), on the CPU."""
    g = torch.Generator().manual_seed(seed)
    v = x.float().cpu()
    sign = torch.where(v < 0, -1.0, 1.0)
    if kind == "wide":
        v = v * torch.pow(2.0, torch.randint(-10, 6, v.shape, generator=g).float())
    else:
        v = sign * torch.pow(2.0, torch.randint(-6, 3, v.shape, generator=g).float())
        v = torch.where(torch.rand(v.shape, generator=g) < 0.03, sign * 2.0**10, v)
    return v.to(F16).to(DEVICE)


# (N, K): projections of a transformer, and convolutions as GEMMs (K = 7 * C, a pointwise one); M from one row to many
TORCH_NK = [(1024, 1024), (3072, 1024), (1024, 3072), (4096, 1024), (1536, 7168), (768, 5376), (96, 672), (1024, 4096), (512, 1536)]
TORCH_M = (1, 32, 64, 100, 320, 1000, 2048)


def torch_pin(m, n, k, bias, layout):
    """The ``"torch"`` posture with torch's default reduction (f16 partial sums allowed) and torch's workspace limit for the op."""
    return L.pin_algo(
        m, n, k, bias, heuristic="torch", allow_f16_reduction=True, form=L.Form("f16", "f16", layout),
        workspace_bytes=L.TORCH_LT_WORKSPACE_BYTES if bias else L.DEFAULT_WORKSPACE_BYTES,
    )  # fmt: skip


@pytest.mark.parametrize("bias", [False, True], ids=["mm", "addmm"])
@pytest.mark.parametrize("layout", ["TN", "NN"])
def test_the_torch_posture_pins_what_torch_runs_in_f16(layout, bias):
    """``torch.mm`` (``torch.addmm``) of f16 tensors in both layouts is the pinned result byte for byte."""
    form = L.Form("f16", "f16", layout)
    misses, runs = [], 0
    for m in TORCH_M:
        for n, k in TORCH_NK:
            pin = torch_pin(m, n, k, bias, layout)
            for seed, kind in enumerate(("normal", "wide", "traps")):
                a, b, c = operands(m, n, k, layout, bias, seed=seed * 11 + m + n)
                if kind != "normal":
                    a, b = (special(x, kind, seed * 7 + m + i) for i, x in enumerate((a, b)))
                ours = run(a, b, c, pin, form, allow=True)
                theirs = torch_of(a, b, c, layout)
                torch.cuda.synchronize()
                runs += 1
                if not same_bytes(ours, theirs):
                    misses.append((m, n, k, kind, L.algo_info(pin.algo_hex)))
    assert runs == len(TORCH_M) * len(TORCH_NK) * 3
    assert not misses, f"{len(misses)} of {runs} runs are not torch's bytes: {misses[:5]}"


@pytest.mark.parametrize("layout", ["TN", "NN"])
@pytest.mark.parametrize("out", ["f16", "f32"])
@pytest.mark.parametrize("bias", [False, True])
def test_the_default_posture_is_the_f64_reference_within_one_accumulation_and_one_rounding(layout, out, bias):
    form = L.Form("f16", out, layout)
    tout = F32 if out == "f32" else F16
    for m, n, k in [(32, 1024, 1024), (320, 3072, 1024), (1, 1536, 7168), (2048, 96, 672), (64, 1000, 1001)]:
        pin = L.pin_algo(m, n, k, bias, form=form, workspace_bytes=64 << 20)
        a, b, c = operands(m, n, k, layout, bias, out=tout, seed=m + k)
        got = run(a, b, c, pin, form).double().cpu()
        a64, b64 = a.double().cpu(), b.double().cpu()
        rhs = b64.t() if layout == "TN" else b64
        exact = a64 @ rhs + (c.double().cpu() if bias else 0)
        absolute = a64.abs() @ rhs.abs() + (c.double().cpu().abs() if bias else 0)
        gamma = (k + 1) * 2.0**-24 / (1 - (k + 1) * 2.0**-24)
        half_ulp = (2.0**-11 if out == "f16" else 2.0**-24) * exact.abs() + (2.0**-25 if out == "f16" else 2.0**-150)
        excess = ((got - exact).abs() - gamma * absolute - half_ulp * 1.0001).max().item()
        assert excess <= 0, (m, n, k, layout, out, bias, excess)


def test_a_workspace_of_64_mib_is_a_limit_the_heuristic_works_under():
    form = L.Form("f16", "f16", "NN")
    pin = L.pin_algo(64, 1536, 7168, True, form=form, workspace_bytes=64 << 20)
    assert pin.workspace_limit == 64 << 20 and pin.workspace_required <= 64 << 20
    assert L.check_algo(64, 1536, 7168, True, pin.algo_hex, workspace_bytes=64 << 20, form=form) == pin.workspace_required


def test_an_algorithm_encodes_its_types_and_a_pin_of_one_form_is_refused_for_another():
    f16_tn = L.Form("f16", "f16", "TN")
    pin = L.pin_algo(64, 1024, 1024, False, form=f16_tn)
    info = L.algo_info(pin.algo_hex)
    assert L.encode_algo(info, form=f16_tn) == pin.algo_hex
    assert L.encode_algo(info, form=L.Form("f16", "f16", "NN")) == pin.algo_hex  # the layout is not in the bytes
    assert L.encode_algo(info) != pin.algo_hex  # bf16 types are
    with pytest.raises(TilefishRuntimeError, match="is not one cuBLASLt produces for bf16 with f32 accumulation: .*bytes are not those"):
        L.check_algo(64, 1024, 1024, False, pin.algo_hex, workspace_bytes=pin.workspace_limit)
    with pytest.raises(TilefishRuntimeError, match="is not one cuBLASLt produces for f16 operands and an f32 result"):
        L.check_algo(64, 1024, 1024, False, pin.algo_hex, workspace_bytes=pin.workspace_limit, form=L.Form("f16", "f32"))


def f16_split(m, n, k, bias, layout, scheme):
    """A split-K of the torch posture's algorithm whose partial sums are added in f16 (``scheme``: 1 INPLACE, 4 OUTPUT_TYPE), or ``None``."""
    form = L.Form("f16", "f16", layout)
    base = L.algo_info(torch_pin(m, n, k, bias, layout).algo_hex)
    for splitk in (2, 3, 4):
        algo = L.encode_algo({**base, "splitk_num": splitk, "reduction_scheme": scheme}, form=form)
        try:
            L.check_algo(m, n, k, bias, algo, workspace_bytes=L.DEFAULT_WORKSPACE_BYTES, allow_f16_reduction=True, form=form)
        except TilefishRuntimeError:
            continue
        return algo
    return None


@pytest.mark.parametrize("scheme", [1, 4])
def test_f16_partial_sums_run_only_with_the_opt_in(scheme):
    m, n, k, layout = 32, 1024, 4096, "NN"
    form = L.Form("f16", "f16", layout)
    algo = f16_split(m, n, k, False, layout, scheme)
    if algo is None:
        pytest.skip(f"cuBLASLt takes no split-K with reduction scheme {scheme} for {m}x{n}x{k} on this device")
    with pytest.raises(TilefishRuntimeError, match=f"reduction scheme {scheme} adds the split-K partial sums in f16, which rounds more than once.*allow_f16_reduction=True"):
        L.check_algo(m, n, k, False, algo, workspace_bytes=L.DEFAULT_WORKSPACE_BYTES, form=form)
    a, b, _ = operands(m, n, k, layout, False, seed=3)
    pin = L.Pinned(algo, 0, L.DEFAULT_WORKSPACE_BYTES)
    first, second = run(a, b, None, pin, form, allow=True), run(a, b, None, pin, form, allow=True)
    torch.cuda.synchronize()
    assert same_bytes(first, second)


# -- the library op: jit, replay, plans, imported tables -------------------------------------------------------------------------------------

M = Dim("M", buckets=[32, 320])


def declarations(**kw):
    """f16 GEMMs of each kind, over the buckets of ``M``: TN without a bias, NN with one, an f32 result with an f32 bias (64 MiB of workspace)."""

    @tilefish.library("f16_tn", acc=f32, impl="cublaslt.gemm", **kw)
    def tn(a: Tensor[M, 1024, f16], b: Tensor[3072, 1024, f16]) -> Tensor[M, 3072, f16]: ...

    @tilefish.library("f16_nn_bias", acc=f32, impl="cublaslt.gemm_bias", layout="NN", **kw)
    def nn(a: Tensor[M, 7168, f16], b: Tensor[7168, 1536, f16], bias: Tensor[1536, f16]) -> Tensor[M, 1536, f16]: ...

    @tilefish.library("f16_to_f32", acc=f32, impl="cublaslt.gemm_bias", workspace_bytes=64 << 20, **kw)
    def wide(a: Tensor[M, 3072, f16], b: Tensor[1024, 3072, f16], bias: Tensor[1024, f32]) -> Tensor[M, 1024, f32]: ...

    return {"tn": (tn, 3072, 1024, "TN", False, F16), "nn": (nn, 1536, 7168, "NN", True, F16), "wide": (wide, 1024, 3072, "TN", True, F32)}


def test_jit_and_replay_agree_for_f16_f32_results_and_both_layouts(tmp_path):
    ops = declarations()
    manifest = runtime.export([op for op, *_ in ops.values()], tmp_path)
    entries = {e["stem"]: e for e in manifest.library_entries}
    assert entries["f16_nn_bias.M32"]["transa"] == "N" and entries["f16_nn_bias.M32"]["lda"] == 1536
    assert entries["f16_to_f32.M320"]["dtypes"] == {"a": "f16", "b": "f16", "bias": "f32", "out": "f32"}
    assert entries["f16_to_f32.M320"]["workspace_bytes"] == 64 << 20
    for op, n, k, layout, bias, out in ops.values():
        for m in (32, 320):
            a, b, c = operands(m, n, k, layout, bias, out=out, seed=m + n)
            args = (a, b, c) if bias else (a, b)
            jit = tilefish.run(op, *args)
            replay = tilefish.run(op, *args, mode="replay", manifest=tmp_path, pin=manifest.digest)
            ref = tilefish.run(op, *args, mode="reference")
            assert jit.dtype == out and same_bytes(jit, replay), (op.name, m)
            assert (jit.double() - ref.double()).abs().max() <= 2**-8 * ref.double().abs().max() + 1e-3, (op.name, m)


def test_the_torch_posture_of_a_declaration_is_torchs_in_both_layouts(tmp_path):
    ops = declarations(heuristic="torch", allow_f16_reduction=True)
    manifest = runtime.export([ops["tn"][0], ops["nn"][0]], tmp_path)
    for key in ("tn", "nn"):
        op, n, k, layout, bias, _ = ops[key]
        for m in (32, 320):
            a, b, c = operands(m, n, k, layout, bias, seed=m)
            args = (a, b, c) if bias else (a, b)
            replay = tilefish.run(op, *args, mode="replay", manifest=tmp_path, pin=manifest.digest)
            assert same_bytes(replay, torch_of(a, b, c, layout)), (op.name, m)


def test_an_imported_f16_table_of_the_layout_nn_runs_the_heuristics_algorithm(tmp_path):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability()
    form = L.Form("f16", "f16", "NN")
    rows = {(m, 1536, 7168, True, "NN"): L.algo_info(L.pin_algo(m, 1536, 7168, True, form=form).algo_hex) for m in (32, 320)}
    table = tilefish.AlgoTable(f"sm_{major * 10 + minor}", props.multi_processor_count, L.version(), rows)
    plain = declarations()["nn"][0]

    @tilefish.library("f16_nn_bias", acc=f32, impl="cublaslt.gemm_bias", layout="NN", algos=table)
    def imported(a: Tensor[M, 7168, f16], b: Tensor[7168, 1536, f16], bias: Tensor[1536, f16]) -> Tensor[M, 1536, f16]: ...

    m_imported = runtime.export([imported], tmp_path / "imported")
    m_plain = runtime.export([plain], tmp_path / "plain")
    for m in (32, 320):
        stem = f"f16_nn_bias.M{m}"
        assert m_imported.library_entry(stem, table.arch)["algo"] == m_plain.library_entry(stem, table.arch)["algo"]
        assert m_imported.library_entry(stem, table.arch)["algo_source"] == "imported"
        a, b, c = operands(m, 1536, 7168, "NN", True, seed=m)
        assert same_bytes(
            tilefish.run(imported, a, b, c, mode="replay", manifest=tmp_path / "imported", pin=m_imported.digest),
            tilefish.run(plain, a, b, c, mode="replay", manifest=tmp_path / "plain", pin=m_plain.digest),
        )


def test_a_plan_of_f16_gemms_runs_the_same_in_jit_replay_and_reference(tmp_path):
    """A graph of an NN GEMM with an f16 bias into a TN GEMM with an f32 result: the arena holds the f16 intermediate and the 64 MiB workspace;
    jit and replay are byte for byte the same, and the reference (torch in f32, rounded once per GEMM) is close."""
    nn = declarations()["nn"][0]

    @tilefish.library("f16_proj_f32", acc=f32, impl="cublaslt.gemm_bias", workspace_bytes=64 << 20)
    def proj(a: Tensor[M, 1536, f16], b: Tensor[1024, 1536, f16], bias: Tensor[1024, f32]) -> Tensor[M, 1024, f32]: ...

    @tilefish.graph(inputs=("x",), weights=("w1", "b1", "w2", "b2"))
    def block(x: Tensor[M, 7168, f16], w1: Tensor[7168, 1536, f16], b1: Tensor[1536, f16], w2: Tensor[1024, 1536, f16], b2: Tensor[1024, f32]):
        return proj(nn(x, w1, b1), w2, b2)

    plan = runtime.export(block, tmp_path)
    for m in (32, 320):
        x, w1, b1 = operands(m, 1536, 7168, "NN", True, seed=m)
        _, w2, b2 = operands(m, 1024, 1536, "TN", True, out=F32, seed=m + 5)
        args = (x, w1, b1, w2, b2)
        jit = tilefish.run(block, *args)
        replay = tilefish.run(block, *args, mode="replay", plan=tmp_path, pin=plan.digest)
        ref = tilefish.run(block, *args, mode="reference")
        assert jit.dtype == F32 and same_bytes(jit, replay), m
        direct = tilefish.run(proj, tilefish.run(nn, x, w1, b1), w2, b2)
        assert same_bytes(jit, direct), m  # the plan runs the pinned GEMMs as they run alone
        assert (jit.double() - ref.double()).abs().max() <= 2**-8 * ref.double().abs().max() + 1e-2, m
