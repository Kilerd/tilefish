"""``cublaslt.matmul`` on a GPU: a GEMM with a cuBLASLt epilogue, its verification per pin and its canary (RFC 0003, milestone E2).

Checked here:

* **X1 of RFC 0003 as a test**: 14 shapes (``M`` 32, 128, 640, 2048 by ``N x K`` of 1536 x 512, 3072 x 1024, 1024 x 3072, and 32 x 4096 x 4096,
  100 x 1000 x 1000), bf16 and f16, both postures (``"tilefish"``, ``"torch"``), and the seven cases ``none`` with ``c``, ``bias``, ``relu``,
  ``relu_bias``, and those three with ``c``. Every pin is verified when it is made; the test checks the record again on operand sets of its own,
  and holds the outcome to what was measured on a B300 MIG slice (sm_103): every pin measured-exact under cuBLASLt 12.8.4; under 13.1.0 the pins of
  algorithm 21 at 32 x 1024 x 3072 that add a bias to ``c`` or take a ReLU of a NaN are binary-defined, the others measured-exact. Elsewhere (another
  architecture or version) the record is checked, and the outcome reported.
* **The rule of split-K**: no epilogue but ``none`` splits ``k`` without the opt-in; on sm_120 (``SM120``, run by a later gate), f16 problems with
  ``allow_f16_reduction=True`` that the heuristic answers with split-K pins, each verified.
* ``jit`` and ``replay`` agree byte for byte; ``export`` records a canary, which the replay runs (once per process and device), which refuses a forged
  digest and a first replay inside a capture; the canary's operands made on the device are the host's ``canary_bytes``.
* **Plans** with ``tilefish.fusion(epilogues="library")``: the rewritten plan runs one ``cublaslt.matmul`` step, writes the formula on the pin's
  accumulator rounded once (``U(R_acc(P))``), and its ``jit``, ``replay`` and capture agree; a pin that is not measured-exact is refused at the build.
"""

import json

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Tensor, TilefishManifestError, TilefishPreconditionError, bf16, f16, f32, runtime  # noqa: E402
from tilefish import _canary, _cublaslt, _library, _matmul  # noqa: E402
from tilefish.errors import TilefishBuildError, TilefishRuntimeError  # noqa: E402

DT = {"bf16": bf16, "f16": f16}
TORCH_DT = {"bf16": torch.bfloat16, "f16": torch.float16}
SHAPES = [(m, n, k) for m in (32, 128, 640, 2048) for (n, k) in ((1536, 512), (3072, 1024), (1024, 3072))] + [(32, 4096, 4096), (100, 1000, 1000)]
CASES = [("none", 1), ("bias", 0), ("relu", 0), ("relu_bias", 0), ("bias", 1), ("relu", 1), ("relu_bias", 1)]
# what was measured on a B300 MIG slice (sm_103, 30 SMs): the binary-defined pins per cuBLASLt version, as (m, n, k, epilogue, beta); none else
BINARY_DEFINED_SM103 = {
    120804: set(),
    130100: {(32, 1024, 3072, "relu_bias", 0), (32, 1024, 3072, "bias", 1), (32, 1024, 3072, "relu", 1), (32, 1024, 3072, "relu_bias", 1)},
}


def _cap():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def matmul_op(name, m, n, k, dtype, epilogue, beta, **kw):
    import inspect

    t = DT[dtype]
    params = {"a": Tensor[m, k, t], "b": Tensor[n, k, t]}
    if epilogue in _library.MATMUL_WITH_BIAS:
        params["bias"] = Tensor[n, t]
    if beta:
        params["c"] = Tensor[m, n, t]

    def fn(*args): ...

    fn.__signature__ = inspect.Signature(
        [inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=v) for p, v in params.items()], return_annotation=Tensor[m, n, t]
    )
    return tilefish.library(name, acc=f32, impl="cublaslt.matmul", epilogue=epilogue, beta=beta, **kw)(fn)


def recheck(spec, algo, seed):
    """The pin run on an operand set of the test's own (``seed``, no special values) against the formula on the accumulator of the same configuration:
    the number of elements that differ."""
    ops = {"a": randn((spec.m, spec.k), TORCH_DT[spec.dtype], seed=seed), "b": randn((spec.n, spec.k), TORCH_DT[spec.dtype], seed=seed + 1, scale=spec.k**-0.5)}
    if spec.bias:
        ops["bias"] = randn((spec.n,), TORCH_DT[spec.dtype], seed=seed + 2, scale=0.5)
    if spec.beta:
        ops["c"] = randn((spec.m, spec.n), TORCH_DT[spec.dtype], seed=seed + 3)
    out = torch.empty(spec.m, spec.n, dtype=TORCH_DT[spec.dtype], device="cuda")
    _library.run_gemm(spec, algo, ops, out, torch.cuda.current_stream().cuda_stream, torch.cuda.current_device())
    acc_algo, _ = _matmul._accumulator_algo(spec, algo, torch.cuda.current_device())
    acc = torch.empty(spec.m, spec.n, dtype=torch.float32, device="cuda")
    _matmul._run_f32(spec, acc_algo, ops, acc, torch.cuda.current_stream().cuda_stream, torch.cuda.current_device())
    return _matmul.compare(torch, out, _matmul.reference(spec, acc, ops))[0]


# -- X1 as a test ----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("posture", ["tilefish", "torch"])
@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_every_pin_is_verified_against_its_formula_and_recorded(dtype, posture, capsys):
    cap, version = _cap(), _cublaslt.version()
    binary = set()
    for m, n, k in SHAPES:
        for epilogue, beta in CASES:
            op = matmul_op(f"x1_{m}_{n}_{k}_{dtype}_{epilogue}_{beta}_{posture}", m, n, k, dtype, epilogue, beta, heuristic=posture)
            (spec,) = _library.library_specs(op)
            built = _library.build_library(spec, arch=cap)
            v = built.verification
            assert v["formula"] == _library.formula(epilogue, beta) and v["elements"] == 2 * m * n and v["operand_sets"] == 2
            assert (built.contract == "measured-exact") == (v["differ"] == 0)
            if built.contract == "measured-exact":
                assert recheck(spec, built.algo, seed=7 * m + n) == 0, (m, n, k, epilogue, beta)
            else:
                binary.add((m, n, k, epilogue, beta))
            cfg = _cublaslt.algo_info(built.algo)
            if epilogue != "none":
                assert cfg["splitk_num"] == 1, (m, n, k, epilogue, cfg)  # no split-K with an epilogue, without the opt-in
            if (m, n, k, epilogue, beta) in BINARY_DEFINED_SM103.get(version, set()) and cap == 103:
                assert cfg["id"] == 21, cfg
    with capsys.disabled():
        print(f"\n[matmul X1] sm_{cap}, cuBLASLt {version}, {dtype}, {posture}: {len(SHAPES) * len(CASES) - len(binary)} measured-exact, binary-defined {sorted(binary)}")
    if cap == 103 and version in BINARY_DEFINED_SM103:
        assert binary == BINARY_DEFINED_SM103[version]


def test_a_pin_and_its_verification_are_deterministic():
    op = matmul_op("det", 640, 3072, 1024, "bf16", "relu_bias", 1)
    (spec,) = _library.library_specs(op)
    first, second = (_library.build_library(spec, arch=_cap(), canary=True) for _ in range(2))
    assert first.entry() == second.entry() and first.canary is not None


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="the split-K pins of sm_120 (RTX 5090)")
def test_split_k_pins_with_the_opt_in_are_verified_on_sm120(capsys):
    """On sm_120 cuBLASLt answers f16 problems of few rows with split-K (cuBLASLt 12.8.4: a split-K of the bias epilogue was measured to round twice
    there, ``docs/numerics.md``, section 4): with ``allow_f16_reduction=True`` such pins are taken, and each is verified like any other; without the
    opt-in no pin with an epilogue splits ``k``."""
    found = []
    for m in (32, 64):
        for n, k in ((1024, 3072), (4096, 3072), (3072, 4096), (1024, 1024)):
            for epilogue, beta in (("bias", 0), ("relu_bias", 0), ("none", 1), ("bias", 1)):
                for posture in ("tilefish", "torch"):
                    for opt_in in (False, True):
                        kw = {"allow_f16_reduction": True} if opt_in else {}
                        op = matmul_op(f"sk_{m}_{n}_{k}_{epilogue}_{beta}_{posture}_{opt_in}", m, n, k, "f16", epilogue, beta, heuristic=posture, **kw)
                        (spec,) = _library.library_specs(op)
                        built = _library.build_library(spec, arch=120)
                        cfg = _cublaslt.algo_info(built.algo)
                        if not opt_in and epilogue != "none":
                            assert cfg["splitk_num"] == 1
                        if cfg["splitk_num"] > 1:
                            found.append((m, n, k, epilogue, beta, posture, opt_in, cfg["id"], cfg["splitk_num"], cfg["reduction_scheme"], built.contract))
                        if built.contract == "measured-exact":
                            assert recheck(spec, built.algo, seed=m + n) == 0
    with capsys.disabled():
        print("\n[matmul sm_120 split-K pins]", *found, sep="\n  ")
    assert found, "no pin split k on this device: the test says nothing here"


# -- jit, replay and the canary --------------------------------------------------------------------------------------------------------------

M = Dim("M", buckets=[32, 128])


def bucketed(name, dtype, epilogue, beta, n=1024, k=3072, **kw):
    import inspect

    t = DT[dtype]
    params = {"a": Tensor[M, k, t], "b": Tensor[n, k, t]}
    if epilogue in _library.MATMUL_WITH_BIAS:
        params["bias"] = Tensor[n, t]
    if beta:
        params["c"] = Tensor[M, n, t]

    def fn(*args): ...

    fn.__signature__ = inspect.Signature(
        [inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=v) for p, v in params.items()], return_annotation=Tensor[M, n, t]
    )
    return tilefish.library(name, acc=f32, impl="cublaslt.matmul", epilogue=epilogue, beta=beta, **kw)(fn)


@pytest.fixture(scope="module")
def ops():
    return [
        bucketed("mm_bias", "bf16", "bias", 0),
        bucketed("mm_relu", "f16", "relu", 0),
        bucketed("mm_relu_bias_c", "bf16", "relu_bias", 1, heuristic="torch"),
        bucketed("mm_c", "f16", "none", 1),
        bucketed("mm_nocanary", "bf16", "bias", 1, canary=False),
    ]


@pytest.fixture(scope="module")
def exported(ops, tmp_path_factory):
    directory = tmp_path_factory.mktemp("matmul-manifest")
    return directory, runtime.export(ops, directory)


def args_of(op, m, seed):
    sig = op.signature
    out = []
    for i, p in enumerate(sig.params):
        shape = [m if str(d) == "M" else d.folded().value for d in p.type.shape]
        scale = shape[-1] ** -0.5 if p.name == "b" else 1.0
        out.append(randn(tuple(shape), TORCH_DT[p.type.dtype.name], seed=seed + i, scale=scale))
    return out


def test_jit_and_replay_agree_and_the_entries_have_their_record_and_canary(ops, exported):
    directory, manifest = exported
    doc = json.loads((directory / "manifest.json").read_text())
    assert len(doc["library"]) == len(ops) * 2
    for e in doc["library"]:
        _library.validate_entry(e)
        assert e["impl"] == "cublaslt.matmul" and e["contract"] in _library.CONTRACTS
        assert ("canary" in e) == (e["op"] != "mm_nocanary")
    for op in ops:
        for m in (32, 128):
            args = args_of(op, m, seed=m)
            jit = tilefish.run(op, *args)
            replay = tilefish.run(op, *args, mode="replay", manifest=directory, pin=manifest.digest)
            assert same_bytes(jit, replay), (op.name, m)
            want = _library.reference_of(op, dict(zip([p.name for p in op.signature.params], args)))
            assert (jit.float() - want.float()).abs().max() <= 0.02 * want.float().abs().max() + 1e-3  # the formula, within the accumulation order


def test_the_canary_is_made_on_the_device_from_the_bytes_the_host_makes():
    for dtype in ("bf16", "f16"):
        for tag, elements in ((0, 1000), (2, 7), (3, 4099)):
            buf = torch.zeros(elements * 2 + 16, dtype=torch.uint8, device="cuda")
            _canary.fill(buf.data_ptr(), elements, dtype, 1, tag, torch.cuda.current_stream().cuda_stream, torch.cuda.current_device())
            torch.cuda.synchronize()
            assert bytes(buf[: elements * 2].cpu().numpy()) == _canary.canary_bytes(dtype, 1, tag, elements)


def test_a_replay_runs_the_canary_once_and_refuses_a_forged_one(ops, exported, tmp_path):
    directory, manifest = exported
    op = ops[0]
    args = args_of(op, 32, seed=3)
    entry = manifest.library_entry("mm_bias.M32", f"sm_{_cap()}")
    (spec,) = [s for s in _library.library_specs(op) if s.stem == "mm_bias.M32"]
    assert _matmul.canary_digest(spec, entry["algo"], 1, torch.cuda.current_device()) == entry["canary"]["out_digest"]
    _matmul._canaries_passed.clear()
    tilefish.run(op, *args, mode="replay", manifest=directory, pin=manifest.digest)
    assert len(_matmul._canaries_passed) == 1
    tilefish.run(op, *args, mode="replay", manifest=directory, pin=manifest.digest)
    assert len(_matmul._canaries_passed) == 1  # once per process and device
    # a forged digest (another build of the same cuBLASLt version would give one): refused before the pin is used
    doc = json.loads((directory / "manifest.json").read_text())
    for e in doc["library"]:
        if e["stem"] == "mm_bias.M32":
            e["canary"]["out_digest"] = "0" * 32
    forged = tmp_path / "forged"
    forged.mkdir()
    (forged / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(
        TilefishManifestError,
        match=r"the canary of the cuBLASLt matmul 'mm_bias.M32' \(sm_\d+, algorithm id \d+ .*, epilogue bias\) gives out with digest [0-9a-f]{32} here, and the manifest "
        r"recorded 0{32} \(seed 1\): .* so it is refused",
    ):
        tilefish.run(op, *args, mode="replay", manifest=forged, trust_directory=True)


def test_a_first_replay_inside_a_capture_is_refused(ops, exported):
    directory, manifest = exported
    op = ops[1]
    args = args_of(op, 128, seed=5)
    _matmul._canaries_passed.clear()
    runtime._pins.clear()
    g = torch.cuda.CUDAGraph()
    with pytest.raises(TilefishRuntimeError, match="runs its canary at its first replay in the process, and this one is inside a CUDA graph capture"):
        with torch.cuda.graph(g):
            tilefish.run(op, *args, mode="replay", manifest=directory, pin=manifest.digest)
    tilefish.run(op, *args, mode="replay", manifest=directory, pin=manifest.digest)  # outside: it runs, and passes


def test_the_operands_of_a_call_are_checked(ops):
    op = ops[2]
    args = args_of(op, 32, seed=1)
    bad = list(args)
    bad[3] = args[3][:, :512].contiguous()  # a c of another shape
    with pytest.raises(TilefishPreconditionError):
        tilefish.run(op, *bad)
    bad = list(args)
    bad[2] = args[2].float()  # an f32 bias
    with pytest.raises(Exception, match="bias"):
        tilefish.run(op, *bad)


def test_a_declaration_that_requires_measured_exact_refuses_another_pin(monkeypatch):
    op = bucketed("strict", "bf16", "bias", 1, contract="measured-exact")
    spec = _library.library_specs(op)[0]
    record = {"formula": "(acc + c) + bias", "accumulator": "f32-form", "operand_sets": 2, "elements": 10, "differ": 3, "max_ulp": 7}
    monkeypatch.setattr(_matmul, "verify", lambda spec, algo, index: ("binary-defined", record))
    with pytest.raises(TilefishBuildError, match=r"asks for contract='measured-exact', and the pinned algorithm id \d+ .* computes bias with beta=1 as no rounding of"):
        _library.build_library(spec, arch=_cap())


# -- plans -----------------------------------------------------------------------------------------------------------------------------------


@tilefish.library("up", acc=f32, impl="cublaslt.gemm")
def up(a: Tensor[M, 3072, bf16], b: Tensor[1024, 3072, bf16]) -> Tensor[M, 1024, bf16]: ...


@tilefish.library("wide", acc=f32, impl="cublaslt.gemm")
def wide(a: Tensor[M, 1024, bf16], b: Tensor[3072, 1024, bf16]) -> Tensor[M, 3072, bf16]: ...


@tilefish.library("up_b", acc=f32, impl="cublaslt.gemm_bias")
def up_b(a: Tensor[M, 1024, f16], b: Tensor[3072, 1024, f16], bias: Tensor[3072, f16]) -> Tensor[M, 3072, f16]: ...


def _relu(h):
    return tilefish.where(h <= 0.0, 0.0, h)


def residual_block(mode):
    @tilefish.graph(fusion=tilefish.fusion(epilogues=mode))
    def block(x: Tensor[M, 1024, bf16], w: Tensor[3072, 1024, bf16], bias: Tensor[3072, bf16], r: Tensor[M, 3072, bf16]):
        return _relu((wide(x, w).to(f32) + r.to(f32)) + bias.to(f32)).to(bf16)

    return block


def residual_block_1024(mode):
    """At 32 x 1024 x 3072 cuBLASLt 13.1.0 pins algorithm 21 on sm_103, which adds c and the bias in neither order (X6)."""

    @tilefish.graph(fusion=tilefish.fusion(epilogues=mode))
    def block_1024(x: Tensor[M, 3072, bf16], w: Tensor[1024, 3072, bf16], bias: Tensor[1024, bf16], r: Tensor[M, 1024, bf16]):
        return _relu((up(x, w).to(f32) + r.to(f32)) + bias.to(f32)).to(bf16)

    return block_1024


def biased_relu(mode):
    @tilefish.graph(fusion=tilefish.fusion(epilogues=mode))
    def ffn(x: Tensor[M, 1024, f16], w: Tensor[3072, 1024, f16], bias: Tensor[3072, f16]):
        return _relu(up_b(x, w, bias).to(f32)).to(f16)

    return ffn


def _accumulator_reference(graph, m, args):
    """``U(R_acc(P))`` of a rewritten plan: the formula on the accumulator of the pin of its one step (the f32 form of its configuration), from the
    plan's own operands, rounded once."""
    plan = tilefish.plan(graph)
    (step,) = plan.steps(M=m)
    spec = plan.callees[step["entry"]].by_stem[step["entry"]]
    algo = runtime.jit_pin(spec, torch.cuda.current_device()).algo
    names = [a["name"] for a in step["args"]][:-1]
    bindings = [a["binding"] for a in step["args"]][:-1]
    params = [p.name for p in graph.signature.params]
    ops = {n: args[params.index(b)] for n, b in zip(names, bindings)}
    acc_algo, _ = _matmul._accumulator_algo(spec, algo, torch.cuda.current_device())
    acc = torch.empty(spec.m, spec.n, dtype=torch.float32, device="cuda")
    _matmul._run_f32(spec, acc_algo, ops, acc, torch.cuda.current_stream().cuda_stream, torch.cuda.current_device())
    return _matmul.reference(spec, acc, ops)


@pytest.mark.parametrize("make", [residual_block, biased_relu], ids=["residual_block", "biased_relu"])
def test_a_rewritten_plan_writes_the_formula_on_the_accumulator_in_jit_replay_and_capture(make, tmp_path):
    lib, exact = make("library"), make("exact")
    plan = tilefish.plan(lib)
    assert [s["kind"] for s in plan.steps(M=32)] == ["library"] and len(tilefish.plan(exact).steps(M=32)) == 2
    written = runtime.export(lib, tmp_path / "lib")
    doc = json.loads((tmp_path / "lib" / "manifest.json").read_text())
    assert {e["contract"] for e in doc["library"]} == {"measured-exact"} and all("canary" in e for e in doc["library"])
    report = json.loads((tmp_path / "lib" / f"{lib.graph.name}.fusion.json").read_text())
    assert report["library_epilogues"][0]["matched"] and report["fusion"]["epilogues"] == "library"
    for m in (32, 128):
        sig = lib.signature
        args = []
        for i, p in enumerate(sig.params):
            shape = tuple(m if str(d) == "M" else d.folded().value for d in p.type.shape)
            args.append(randn(shape, TORCH_DT[p.type.dtype.name], seed=10 * m + i, scale=(shape[-1] ** -0.5 if p.name == "w" else 1.0)))
        got = tilefish.run(lib, *args)
        replay = tilefish.run(lib, *args, mode="replay", plan=tmp_path / "lib", pin=written.digest)
        assert same_bytes(got, replay)
        assert same_bytes(got, _accumulator_reference(lib, m, args)), m  # U(R_acc(P)) exactly: the measured-exact pin
        unfused = tilefish.run(exact, *args)  # U(P): the GEMM rounds before the epilogue
        differ = int((got.view(torch.int16) != unfused.view(torch.int16)).sum())
        if make is biased_relu:  # the bias was already in the GEMM, and ReLU commutes with the rounding (X1): the bytes of U(P), contract (i)
            assert differ == 0
        else:
            assert differ > 0.05 * got.numel(), differ  # one rounding fewer: about 15 to 30 % of the elements (X1)
        # the plan captured in a CUDA graph (the canary ran at the replay above, before the capture) replays to the same bytes
        captured = tilefish.capture(lib, *args, mode="replay", plan=tmp_path / "lib", pin=written.digest)
        captured.replay()
        torch.cuda.synchronize()
        (out,) = captured.outputs if isinstance(captured.outputs, (tuple, list)) else (captured.outputs,)
        assert same_bytes(out, got)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3), reason="what was measured on sm_103")
def test_a_rewrite_whose_pin_is_not_measured_exact_is_refused(tmp_path):
    """Under cuBLASLt 13.1.0 on sm_103 the pin of a residual add with a bias at 32 x 1024 x 3072 is algorithm 21, which adds them in neither order
    (X6 of RFC 0003): the rewrite declares contract="measured-exact", so its export is refused (and a barrier keeps the GEMM unfused)."""
    g = residual_block_1024("library")
    if _cublaslt.version() != 130100:
        runtime.export(g, tmp_path / "ok")  # 12.8.4: every pin is measured-exact
        return
    with pytest.raises(TilefishBuildError, match=r"cannot pin 'up__relu_bias_c.M32' .* asks for contract='measured-exact'"):
        runtime.export(g, tmp_path / "refused")
