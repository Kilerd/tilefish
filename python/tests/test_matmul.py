"""``cublaslt.matmul``: a GEMM with a cuBLASLt epilogue (RFC 0003, milestone E2), declared, specified, written to a manifest, read back and planned,
with no GPU.

``@tilefish.library("up", acc=f32, impl="cublaslt.matmul", epilogue="relu_bias", beta=1)`` over ``(a, b, bias, c) -> out`` is
``out = relu((acc + c) + bias)`` on the f32 accumulator, rounded once: the library's bytes under the pin, each pin verified at the build against that
formula and recorded ``measured-exact`` or ``binary-defined`` (``docs/manifest-schema-3.md``, section 9.5). ``tilefish.fusion(epilogues="library")``
rewrites a GEMM and the arithmetic after it into one. What the GPU does (the pins, the verification, the canary, the plans' bytes) is
``test_gpu_matmul.py``.
"""

import copy
import json

import pytest

import tilefish
from tilefish import Dim, Manifest, Tensor, TilefishDeclarationError, TilefishManifestError, TilefishPlanError, bf16, f16, f32
from tilefish import _cublaslt, _fusion, _library, _matmul
from tilefish._library import BuiltLibrary, built_from_manifest, formula, library_specs, validate_entry
from tilefish._manifest import manifest_digest
from tilefish.aot import write_manifest

M = Dim("M", buckets=[32, 64])
ALGO = "0123456789abcdef" * 8
VERSION = 120804
SMS = 30
EXACT = {"formula": None, "accumulator": "f32-form", "operand_sets": 2, "elements": 4096, "differ": 0, "max_ulp": 0}


def matmul_op(epilogue="relu_bias", beta=1, dtype=bf16, name="mm", **kw):
    """A cublaslt.matmul of [M, 256] by [512, 256], with the operands its epilogue and beta take."""
    params = {"a": Tensor[M, 256, dtype], "b": Tensor[512, 256, dtype]}
    if epilogue in ("bias", "relu_bias"):
        params["bias"] = Tensor[512, dtype]
    if beta:
        params["c"] = Tensor[M, 512, dtype]
    return declare(params, Tensor[M, 512, dtype], name, impl="cublaslt.matmul", epilogue=epilogue, beta=beta, **kw)


def declare(params, ret, name, **kw):
    import inspect

    def fn(*args): ...

    fn.__signature__ = inspect.Signature(
        [inspect.Parameter(k, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=v) for k, v in params.items()], return_annotation=ret
    )
    return tilefish.library(name, acc=f32, **kw)(fn)


def built(spec, contract="measured-exact", differ=0, canary=True, **kw):
    v = dict(EXACT, formula=formula(spec.epilogue, spec.beta), differ=differ, max_ulp=3 if differ else 0)
    made = {"seed": 1, "out_digest": "0123456789abcdef" * 2} if canary else None
    return BuiltLibrary(spec, kw.pop("arch", "sm_103"), kw.pop("algo", ALGO), kw.pop("version", VERSION), kw.pop("sm_count", SMS), contract, v, made)


def write(tmp_path, *lib, **kw):
    write_manifest([], tmp_path / "m" / "manifest.json", cubins=True, library=lib, **kw)
    return Manifest.load(tmp_path / "m")


# -- the declaration ---------------------------------------------------------------------------------------------------------------------------


def test_a_matmul_declares_its_epilogue_beta_canary_and_contract():
    op = matmul_op()
    assert (op.impl, op.epilogue, op.beta, op.canary, op.contract) == ("cublaslt.matmul", "relu_bias", 1, True, None)
    assert op.workspace_bytes == 8_519_680 and op.heuristic == "tilefish"
    plain = matmul_op("none", 1, heuristic="torch")
    assert plain.workspace_bytes == 8_519_680  # torch runs addmm with a 2-D c as cublasGemmEx: its cuBLAS workspace
    assert matmul_op("relu", 0, heuristic="torch").workspace_bytes == 1_048_576  # an epilogue: torch's cuBLASLt workspace
    assert matmul_op("bias", 0, canary=False, contract="measured-exact").contract == "measured-exact"


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"epilogue": None}, r"needs epilogue= one of \['none', 'bias', 'relu', 'relu_bias'\]"),
        ({"epilogue": "gelu_bias"}, "GELU is not offered yet"),
        ({"beta": 2}, "beta must be 0 \\(no c\\) or 1"),
        ({"beta": True}, "beta must be 0"),
        ({"canary": 1}, "canary must be True"),
        ({"contract": "binary-defined"}, "contract must be None"),
        ({"algos": _library.AlgoTable("sm_103", 30, 120804, {(32, 512, 256, True): ALGO})}, "does not import algorithms"),
    ],
)
def test_the_options_of_a_matmul_are_checked(kw, message):
    with pytest.raises(TilefishDeclarationError, match=message):
        matmul_op(**{"epilogue": "bias", "beta": 0, **kw})


def test_the_options_of_a_matmul_need_its_impl():
    with pytest.raises(TilefishDeclarationError, match="epilogue, beta, canary and contract need impl='cublaslt.matmul'"):
        declare({"a": Tensor[M, 256, bf16], "b": Tensor[512, 256, bf16]}, Tensor[M, 512, bf16], "g", impl="cublaslt.gemm", epilogue="relu")
    with pytest.raises(TilefishDeclarationError, match="need impl='cublaslt.matmul'"):
        tilefish.library("x", beta=1)


def test_the_signature_has_the_operands_of_the_epilogue_and_beta():
    a, b = Tensor[M, 256, bf16], Tensor[512, 256, bf16]
    with pytest.raises(TilefishDeclarationError, match=r"with epilogue='relu_bias' and beta=0 takes 3 input tensors \(a: Tensor\[M, K, T\], b: Tensor\[N, K, T\], bias: Tensor\[N, T\]\)"):
        declare({"a": a, "b": b}, Tensor[M, 512, bf16], "g", impl="cublaslt.matmul", epilogue="relu_bias")
    with pytest.raises(TilefishDeclarationError, match=r"takes 3 input tensors .*c: Tensor\[M, N, T\]"):
        declare({"a": a, "b": b}, Tensor[M, 512, bf16], "g", impl="cublaslt.matmul", epilogue="relu", beta=1)
    with pytest.raises(TilefishDeclarationError, match="takes bf16 or f16 operands and returns their type"):
        declare({"a": a, "b": b}, Tensor[M, 512, f32], "g", impl="cublaslt.matmul", epilogue="relu")
    with pytest.raises(TilefishDeclarationError, match="takes bf16 or f16 operands"):
        declare({"a": Tensor[M, 256, f32], "b": Tensor[512, 256, f32]}, Tensor[M, 512, f32], "g", impl="cublaslt.matmul", epilogue="relu")
    with pytest.raises(TilefishDeclarationError, match="c must be a contiguous bf16 tensor of rank 2"):
        declare({"a": a, "b": b, "c": Tensor[M, 512, f16]}, Tensor[M, 512, bf16], "g", impl="cublaslt.matmul", epilogue="none", beta=1)
    with pytest.raises(TilefishDeclarationError, match=r"c must be \[M, N\] = \[M, 512\]"):
        declare({"a": a, "b": b, "c": Tensor[M, 256, bf16]}, Tensor[M, 512, bf16], "g", impl="cublaslt.matmul", epilogue="none", beta=1)
    with pytest.raises(TilefishDeclarationError, match="bias must be a bf16 tensor of rank 1"):
        declare({"a": a, "b": b, "bias": Tensor[512, f32]}, Tensor[M, 512, bf16], "g", impl="cublaslt.matmul", epilogue="bias")
    with pytest.raises(TilefishDeclarationError, match="allow_f16_reduction is the opt-in of f16 operands"):
        matmul_op("relu", 0, allow_f16_reduction=True)
    matmul_op("relu", 0, dtype=f16, allow_f16_reduction=True, layout="TN", sm_count_target=16)


def test_the_formula_is_the_librarys_order():
    assert [formula(e, b) for b in (0, 1) for e in _library.MATMUL_EPILOGUES] == [
        "acc", "acc + bias", "relu(acc)", "relu(acc + bias)", "acc + c", "(acc + c) + bias", "relu(acc + c)", "relu((acc + c) + bias)",
    ]  # fmt: skip


def test_the_digest_covers_the_epilogue_beta_canary_and_contract():
    digests = {_library.declaration_digest(op) for op in (
        matmul_op("relu_bias", 1), matmul_op("relu_bias", 0), matmul_op("bias", 0), matmul_op("relu_bias", 1, canary=False),
        matmul_op("relu_bias", 1, contract="measured-exact"),
    )}  # fmt: skip
    assert len(digests) == 5
    canonical = _library._canonical(matmul_op("relu", 1))
    assert (canonical["impl"], canonical["epilogue"], canonical["beta"], canonical["canary"]) == ("cublaslt.matmul", "relu", 1, True) and "contract" not in canonical


# -- the entry -----------------------------------------------------------------------------------------------------------------------------------


def test_the_entry_says_the_epilogue_beta_and_the_operands():
    spec = library_specs(matmul_op())[0]
    e = built(spec).entry()
    assert (e["impl"], e["epilogue"], e["beta"], e["contract"]) == ("cublaslt.matmul", "relu_bias", 1, "measured-exact")
    assert e["dtypes"] == {"a": "bf16", "b": "bf16", "bias": "bf16", "c": "bf16", "out": "bf16"}
    assert [(p["name"], p["shape"], p["access"]) for p in e["abi"]["params"]] == [
        ("a", ["M", 256], "in"), ("b", [512, 256], "in"), ("bias", [512], "in"), ("c", ["M", 512], "in"), ("out", ["M", 512], "out"),
    ]  # fmt: skip
    assert e["verification"]["formula"] == "relu((acc + c) + bias)" and e["canary"] == {"seed": 1, "out_digest": "0123456789abcdef" * 2}
    assert set(e) == set(_library.MATMUL_ENTRY_KEYS) | {"canary"}
    validate_entry(e)
    assert "canary" not in built(spec, canary=False).entry()
    plain = built(library_specs(matmul_op("relu", 0))[0]).entry()
    assert plain["dtypes"] == {"a": "bf16", "b": "bf16", "out": "bf16"} and plain["beta"] == 0 and len(plain["abi"]["params"]) == 3
    validate_entry(plain)


def test_gemm_and_gemm_bias_entries_are_what_they_were():
    """The new impl adds nothing to the two of tilefish 0.0.4: their entries and digests are unchanged (the vectors of the Rust reader hold them)."""

    @tilefish.library("g", acc=f32, impl="cublaslt.gemm_bias")
    def g(a: Tensor[M, 256, bf16], b: Tensor[512, 256, bf16], bias: Tensor[512, bf16]) -> Tensor[M, 512, bf16]: ...

    e = BuiltLibrary(library_specs(g)[0], "sm_103", ALGO, VERSION, SMS).entry()
    assert set(e) == set(_library.ENTRY_KEYS) and e["epilogue"] == "bias" and "beta" not in e
    assert "epilogue" not in _library._canonical(g)


def mutated(entry, path, value, also=()):
    e = copy.deepcopy(entry)
    for p, v in [(path, value), *also]:
        target = e
        for key in p[:-1]:
            target = target[key]
        if v is None:
            del target[p[-1]]
        else:
            target[p[-1]] = v
    return e


@pytest.mark.parametrize(
    "path, value, message",
    [
        (["epilogue"], "gelu", r"the epilogue of cublaslt.matmul must be one of"),
        (["beta"], 1.0, "beta must be the int 0 or 1"),
        (["beta"], True, "beta must be the int 0 or 1"),
        (["beta"], 2, "beta must be the int 0 or 1"),
        (["contract"], "exact", "contract must be one of"),
        (["contract"], "binary-defined", "contract is 'binary-defined', but the verification found 0 elements that differ"),
        (["verification", "formula"], "acc + bias", r"verification.formula must be 'relu\(\(acc \+ c\) \+ bias\)'"),
        (["verification", "accumulator"], "torch", "verification.accumulator must be one of"),
        (["verification", "elements"], 0, "verification.elements must be a positive int"),
        (["verification", "differ"], -1, "verification.differ must be an int of at least 0"),
        (["verification", "differ"], 5000, "verification.differ \\(5000\\) is more than verification.elements"),
        (["verification", "max_ulp"], 2, "verification.max_ulp is 2, but no element differs"),
        (["verification", "seed"], 3, "verification must be an object of exactly"),
        (["verification", "max_ulp"], None, "verification must be an object of exactly"),
        (["canary", "seed"], -1, "canary must be"),
        (["canary", "seed"], 1 << 32, "canary must be"),
        (["canary", "out_digest"], "AB" * 16, "canary must be"),
        (["canary", "y_digest"], "ab" * 16, "canary must be"),
        (["canary"], {}, "canary must be"),
        (["algo_source"], "imported", "unknown \\['algo_source'\\]"),
        (["allow_f16_reduction"], True, "allow_f16_reduction is the opt-in of f16 operands"),
        (["allow_f32_reduction"], True, "allow_f32_reduction is the opt-in of f32 operands"),
        (["dtypes", "c"], None, r"dtypes must be \{'a': 'bf16', 'b': 'bf16', 'bias': 'bf16', 'c': 'bf16', 'out': 'bf16'\}"),
        (["dtypes", "out"], "f32", "dtypes must be"),
        (["abi", "params", 3, "dtype"], "f16", r"abi.params\[3\] must be the bf16 pointer #3"),
        (["abi", "params", 3, "shape"], [32, 256], r"abi.params\[3\].shape\[1\] is 256 = 256, but m, n and k say 512"),
        (["beta"], 0, r"verification.formula must be 'relu\(acc \+ bias\)'"),
        (["contract"], None, r"missing \['contract'\]"),
        (["sm_count_target"], 7, "sm_count_target is absent or an int from 8"),
    ],
)
def test_the_reader_refuses_a_matmul_entry_that_breaks_a_rule(path, value, message):
    e = built(library_specs(matmul_op())[0]).entry()
    with pytest.raises(TilefishManifestError, match=message):
        validate_entry(mutated(e, path, value))


def test_the_operands_follow_from_the_epilogue_and_beta():
    e = built(library_specs(matmul_op())[0]).entry()
    no_c = [(["verification", "formula"], "relu(acc + bias)")]
    with pytest.raises(TilefishManifestError, match=r"dtypes must be \{'a': 'bf16', 'b': 'bf16', 'bias': 'bf16', 'out': 'bf16'\}"):
        validate_entry(mutated(e, ["beta"], 0, no_c))
    with pytest.raises(TilefishManifestError, match=r"abi.params must list 4 pointers"):
        validate_entry(mutated(e, ["beta"], 0, [*no_c, (["dtypes", "c"], None)]))
    no_bias = [(["verification", "formula"], "relu(acc + c)")]
    with pytest.raises(TilefishManifestError, match=r"dtypes must be \{'a': 'bf16', 'b': 'bf16', 'c': 'bf16', 'out': 'bf16'\}"):
        validate_entry(mutated(e, ["epilogue"], "relu", no_bias))
    with pytest.raises(TilefishManifestError, match=r"abi.params must list 4 pointers \(a, b, c, out in this order\)"):
        validate_entry(mutated(e, ["epilogue"], "relu", [*no_bias, (["dtypes", "bias"], None)]))


def test_a_binary_defined_entry_records_how_far_it_is():
    e = built(library_specs(matmul_op())[0], contract="binary-defined", differ=17).entry()
    validate_entry(e)
    assert e["verification"]["differ"] == 17 and e["verification"]["max_ulp"] == 3
    with pytest.raises(TilefishManifestError, match="contract is 'measured-exact', but the verification found 17"):
        validate_entry(mutated(e, ["contract"], "measured-exact"))


def test_a_manifest_holds_matmul_entries_and_digests_their_new_fields(tmp_path):
    spec = library_specs(matmul_op())[0]
    m = write(tmp_path, built(spec))
    doc = json.loads((tmp_path / "m" / "manifest.json").read_text())
    for change in (
        lambda e: e.update(contract="binary-defined"),
        lambda e: e["verification"].update(differ=1),
        lambda e: e["canary"].update(out_digest="f" * 32),
        lambda e: e.pop("canary"),
        lambda e: e.update(beta=0),
    ):
        other = copy.deepcopy(doc)
        change(other["library"][0])
        assert manifest_digest(other) != m.digest
    # a re-export with another contract, or another canary, is another binary
    with pytest.raises(TilefishManifestError, match=r"already has a different library entry for 'mm.M32' on sm_103 \(contract differs\)"):
        write(tmp_path, built(spec, contract="binary-defined", differ=4), merge=True)
    with pytest.raises(TilefishManifestError, match=r"\(canary differs\)"):
        b = built(spec)
        write(tmp_path, BuiltLibrary(spec, "sm_103", ALGO, VERSION, SMS, "measured-exact", b.verification, {"seed": 1, "out_digest": "f" * 32}), merge=True)


def test_a_replay_compares_the_entry_with_the_declaration_and_runs_the_canary(tmp_path, monkeypatch):
    monkeypatch.setattr(_cublaslt, "version", lambda: VERSION)
    ran = []
    monkeypatch.setattr(_matmul, "check_canary", lambda b, index: ran.append((b.stem, b.canary["seed"])))
    monkeypatch.setattr(_library, "_device_index", lambda index: 0)
    op = matmul_op()
    m = write(tmp_path, *(built(s) for s in library_specs(op)))
    got = built_from_manifest(m, library_specs(op)[1], "sm_103", SMS)
    assert (got.contract, got.canary["seed"], got.verification["formula"]) == ("measured-exact", 1, "relu((acc + c) + bias)")
    assert got.entry() == m.library_entry("mm.M64", "sm_103") and ran == [("mm.M64", 1)]
    # another epilogue, another beta: not what the declaration asks for
    for other in (matmul_op("bias", 1), matmul_op("relu_bias", 0)):
        with pytest.raises(TilefishManifestError, match="is not what the declaration asks for: first difference at"):
            built_from_manifest(m, library_specs(other)[1], "sm_103", SMS)
    # a declaration that requires measured-exact refuses a binary-defined entry
    strict = matmul_op(contract="measured-exact")
    m2 = write(tmp_path / "b", *(built(s, contract="binary-defined", differ=9) for s in library_specs(strict)))
    with pytest.raises(TilefishManifestError, match=r"is binary-defined, and the declaration asks for contract='measured-exact'"):
        built_from_manifest(m2, library_specs(strict)[0], "sm_103", SMS)
    # without a canary in the entry nothing runs
    ran.clear()
    m3 = write(tmp_path / "c", *(built(s, canary=False) for s in library_specs(op)))
    built_from_manifest(m3, library_specs(op)[0], "sm_103", SMS)
    assert ran == []


# -- the binding ---------------------------------------------------------------------------------------------------------------------------------


def test_the_binding_describes_the_epilogue_and_beta():
    d = _cublaslt.describe(32, 512, 256, False, epilogue="relu_bias", beta=1)
    assert (d["epilogue"], d["cublaslt_epilogue"], d["beta"], d["c_pointer"], d["bias_type"]) == ("relu_bias", 6, 1.0, "C[M, N]", "CUDA_R_16BF")
    assert d["reduction_schemes"] == [0]  # an epilogue: no split-K without the opt-in
    assert _cublaslt.describe(32, 512, 256, False, epilogue="none", beta=1)["reduction_schemes"] == [0, 2]  # beta * C alone keeps split-K
    old = _cublaslt.describe(32, 512, 256, True)  # gemm_bias: as it was
    assert (old["epilogue"], old["beta"]) == ("bias", 0.0) and "cublaslt_epilogue" not in old
    with pytest.raises(tilefish.TilefishRuntimeError, match="epilogue must be one of"):
        _cublaslt.describe(32, 512, 256, False, epilogue="gelu")
    with pytest.raises(tilefish.TilefishRuntimeError, match="beta must be 0 or 1"):
        _cublaslt.describe(32, 512, 256, False, epilogue="relu", beta=2)


def test_an_epilogue_has_no_split_k_without_the_opt_in_in_its_own_words():
    form = _cublaslt.Form()
    assert _cublaslt.reduction_problem(3, 2, False, True, form, "relu") == (
        "split-K count 3 with the relu epilogue may apply it to a sum already rounded to bf16 (as the bias epilogue was measured to, with f32 partial "
        "sums too), which rounds twice; a GEMM with an epilogue has no split-K unless its declaration opts in with allow_bf16_reduction=True"
    )
    assert "with the bias epilogue may add the bias" in _cublaslt.reduction_problem(3, 2, False, True, form)  # gemm_bias's words, unchanged
    assert _cublaslt.reduction_problem(3, 2, True, True, form, "relu_bias") is None
    assert _cublaslt.reduction_problem(3, 2, False, False, form) is None  # DEFAULT (beta * C too): f32 partial sums are taken


def test_the_canary_operands_are_a_b_bias_and_c_in_the_order_of_the_abi():
    spec = library_specs(matmul_op())[0]
    assert _matmul.canary_elements(spec) == [("a", 0, 32 * 256), ("b", 1, 512 * 256), ("bias", 2, 512), ("c", 3, 32 * 512)]
    assert _matmul.canary_elements(library_specs(matmul_op("relu", 0))[0]) == [("a", 0, 32 * 256), ("b", 1, 512 * 256)]


def test_compare_counts_bits_and_measures_ulps():
    torch = pytest.importorskip("torch")
    a = torch.tensor([1.0, 2.0, float("nan"), -0.0, 3.0], dtype=torch.bfloat16)
    b = torch.tensor([1.0, 2.015625, float("nan"), 0.0, float("inf")], dtype=torch.bfloat16)
    assert _matmul.compare(torch, a, a) == (0, 0)
    differ, ulp = _matmul.compare(torch, a, b)
    assert differ == 3 and ulp == 1  # 2.0 vs its next value; -0 vs +0 (one apart); 3.0 vs inf is not finite


# -- plans: tilefish.fusion(epilogues="library") ---------------------------------------------------------------------------------------------


@tilefish.library("up", acc=f32, impl="cublaslt.gemm")
def up(a: Tensor[M, 256, bf16], b: Tensor[512, 256, bf16]) -> Tensor[M, 512, bf16]: ...


@tilefish.library("up_b", acc=f32, impl="cublaslt.gemm_bias", heuristic="torch", allow_bf16_reduction=True)
def up_b(a: Tensor[M, 256, bf16], b: Tensor[512, 256, bf16], bias: Tensor[512, bf16]) -> Tensor[M, 512, bf16]: ...


def relu(h):
    return tilefish.where(h <= 0.0, 0.0, h)


def graph_of(body, mode="library", name="g"):
    def fn(x: Tensor[M, 256, bf16], w: Tensor[512, 256, bf16], bias: Tensor[512, bf16], r: Tensor[M, 512, bf16]):
        return body(x, w, bias, r)

    fn.__name__ = fn.__qualname__ = name
    return tilefish.graph(fusion=tilefish.fusion(epilogues=mode))(fn)


def steps_of(g, m=32):
    return [(s["entry"], s["kind"]) for s in tilefish.plan(g).steps(M=m)]


CASES = [
    ("bias", lambda x, w, b, r: (up(x, w).to(f32) + b.to(f32)).to(bf16), "up__bias"),
    ("bias, the other side", lambda x, w, b, r: (b.to(f32) + up(x, w).to(f32)).to(bf16), "up__bias"),
    ("relu", lambda x, w, b, r: relu(up(x, w).to(f32)).to(bf16), "up__relu"),
    ("relu_bias", lambda x, w, b, r: relu(up(x, w).to(f32) + b.to(f32)).to(bf16), "up__relu_bias"),
    ("residual", lambda x, w, b, r: (up(x, w).to(f32) + r.to(f32)).to(bf16), "up__none_c"),
    ("residual, bias", lambda x, w, b, r: ((up(x, w).to(f32) + r.to(f32)) + b.to(f32)).to(bf16), "up__bias_c"),
    ("residual, relu", lambda x, w, b, r: relu(up(x, w).to(f32) + r.to(f32)).to(bf16), "up__relu_c"),
    ("residual, bias, relu", lambda x, w, b, r: relu((up(x, w).to(f32) + r.to(f32)) + b.to(f32)).to(bf16), "up__relu_bias_c"),
    ("gemm_bias, relu", lambda x, w, b, r: relu(up_b(x, w, b).to(f32)).to(bf16), "up_b__relu_bias"),
]


@pytest.mark.parametrize("label, body, into", CASES, ids=[c[0] for c in CASES])
def test_a_library_epilogue_replaces_the_gemm_and_its_arithmetic(label, body, into):
    g = graph_of(body)
    assert steps_of(g) == [(f"{into}.M32", "library")]
    op = tilefish.plan(g).callees[f"{into}.M32"].op
    assert op.impl == "cublaslt.matmul" and op.contract == "measured-exact" and op.canary
    exact = graph_of(body, "exact")
    assert [e for e, _ in steps_of(exact)][0] in ("up.M32", "up_b.M32") and len(steps_of(exact)) == 2  # the default: the GEMM and a region
    (w,) = _fusion.report(g)["library_epilogues"]
    assert (w["matched"], w["applied"], w["into"]) == (True, True, into)
    assert w["contract"] == ("exact" if label in ("relu", "gemm_bias, relu") else "measured-exact")  # ReLU alone commutes with the rounding
    # planned exact, the report says what the mode would do (the advisor reads it), with the same epilogue
    (c,) = _fusion.report(exact)["library_epilogues"]
    assert (c["matched"], c["applied"], c["epilogue"], c["beta"]) == (True, False, w["epilogue"], w["beta"])


def test_the_rewritten_op_keeps_the_gemms_options():
    g = graph_of(lambda x, w, b, r: relu(up_b(x, w, b).to(f32)).to(bf16))
    op = tilefish.plan(g).callees["up_b__relu_bias.M32"].op
    assert (op.heuristic, op.allow_bf16_reduction, op.workspace_bytes, op.layout) == ("torch", True, up_b.workspace_bytes, "TN")
    assert [p.name for p in op.signature.params] == ["a", "b", "bias"]
    # the same GEMM and epilogue in two graphs is one operator
    other = graph_of(lambda x, w, b, r: relu(up_b(x, w, b).to(f32)).to(bf16), name="other")
    assert tilefish.plan(other).callees["up_b__relu_bias.M32"].op is op


def test_a_plan_step_of_a_rewrite_binds_the_bias_and_c_in_the_order_of_the_abi():
    g = graph_of(lambda x, w, b, r: relu((up(x, w).to(f32) + r.to(f32)) + b.to(f32)).to(bf16))
    (step,) = tilefish.plan(g).steps(M=64)
    assert [(a["name"], a["binding"]) for a in step["args"]] == [("a", "x"), ("b", "w"), ("bias", "bias"), ("c", "r"), ("out", "out0")]


NOT_MATCHED = [
    ("the bias before the residual", lambda x, w, b, r: ((up(x, w).to(f32) + b.to(f32)) + r.to(f32)).to(bf16), "the residual is added after the bias"),
    ("an f32 result", lambda x, w, b, r: (up(x, w).to(f32) + b.to(f32)), "does not end in one plain .to"),
    ("y read twice", lambda x, w, b, r: (lambda y: ((y.to(f32) + b.to(f32)).to(bf16), (y.to(f32) * 2.0).to(bf16)))(up(x, w)), "read by something else"),
    ("a multiply", lambda x, w, b, r: (up(x, w).to(f32) * 2.0).to(bf16), "does not end in one plain .to"),
    ("relu written another way", lambda x, w, b, r: tilefish.where(up(x, w).to(f32) > 0.0, up(x, w).to(f32), 0.0).to(bf16), None),
    ("a gemm_bias with a residual", lambda x, w, b, r: (up_b(x, w, b).to(f32) + r.to(f32)).to(bf16), "the residual is added after the bias"),
    ("nothing to compute", lambda x, w, b, r: up(x, w).to(f32).to(bf16), "nothing between"),
    ("a bias of another type", lambda x, w, b, r: (up(x, w).to(f32) + b.to(f16).to(f32)).to(bf16), "not a dense tensor of bf16"),
]


@pytest.mark.parametrize("label, body, reason", NOT_MATCHED, ids=[c[0] for c in NOT_MATCHED])
def test_arithmetic_that_is_no_epilogue_of_the_library_is_left_as_it_is(label, body, reason):
    g = graph_of(body)
    entries = [e for e, _ in steps_of(g)]
    assert not any("__" in e for e in entries), entries
    assert steps_of(g) == steps_of(graph_of(body, "exact"))  # the plan is the exact one
    if reason is not None:
        assert any(not w["matched"] and reason in w["reason"] for w in _fusion.report(g)["library_epilogues"])


def test_a_barrier_or_a_returned_intermediate_keeps_the_rounding():
    def barriered(x, w, b, r):
        y = tilefish.barrier(up(x, w))
        return (y.to(f32) + b.to(f32)).to(bf16)

    def returned(x, w, b, r):
        y = up(x, w)
        return (y.to(f32) + b.to(f32)).to(bf16), y

    for body in (barriered, returned):
        assert not any("__" in e for e, _ in steps_of(graph_of(body)))


def test_a_residual_made_after_the_gemm_is_not_read_by_its_epilogue():
    def late(x, w, b, r):
        y = up(x, w)
        c = (r.to(f32) * 2.0).to(bf16)
        return (y.to(f32) + c.to(f32)).to(bf16)

    g = graph_of(late)
    assert not any("__" in e for e, _ in steps_of(g))
    assert any("made before the GEMM" in w.get("reason", "") for w in _fusion.report(g)["library_epilogues"])


def test_a_graph_without_a_widened_gemm_result_has_the_report_it_had():
    @tilefish.graph
    def plain(x: Tensor[M, 256, bf16], w: Tensor[512, 256, bf16], bias: Tensor[512, bf16]):
        y = up(x, w)
        return y * 2.0

    assert "library_epilogues" not in _fusion.report(plain)


def test_the_advisor_reports_a_matched_epilogue_as_an_action_of_the_mode(tmp_path):
    from tilefish import _advise

    exact = graph_of(CASES[3][1], "exact", name="advised")
    p = tilefish.plan(exact)
    gemm_step, region_step = (s["entry"] for s in p.steps(M=64))
    manifest = {"schema": 3, "kernels": [{"stem": region_step, "op": region_step.split(".")[0], "arch": "sm_103", "abi": {"params": []}}], "library": [
        {"stem": gemm_step, "op": "up", "arch": "sm_103", "impl": "cublaslt.gemm", "m": 64, "n": 512, "k": 256, "dtypes": {"a": "bf16", "b": "bf16", "out": "bf16"}},
    ]}  # fmt: skip
    doc = dict(p.doc, arch="sm_103")
    advice = _advise.analyse(doc, manifest, report=_fusion.report(exact), bucket={"M": 64})
    (edge,) = [e for e in advice.edges if e.producer == 0]
    assert edge.blockers[0] == "epilogue-library-cublaslt" and "cuBLASLt's relu_bias" in edge.detail[0]
    assert _advise.BLOCKERS["epilogue-library-cublaslt"][1] == "E2"
    # an order the library does not have: still future work, with why
    other = graph_of(NOT_MATCHED[0][1], "exact", name="advised_other")
    p2 = tilefish.plan(other)
    g2, r2 = (s["entry"] for s in p2.steps(M=64))
    manifest["kernels"][0].update(stem=r2, op=r2.split(".")[0])
    advice = _advise.analyse(dict(p2.doc, arch="sm_103"), manifest, report=_fusion.report(other), bucket={"M": 64})
    (edge,) = [e for e in advice.edges if e.producer == 0]
    assert edge.blockers[0] == "epilogue-library" and "the residual is added after the bias" in edge.detail[0]


def test_the_mode_is_checked_and_is_in_the_report_not_in_the_names_of_regions():
    with pytest.raises(TilefishPlanError, match=r"epilogues must be one of \['exact', 'library'\]"):
        tilefish.fusion(epilogues="accumulators")
    assert tilefish.fusion(epilogues="library") != tilefish.fusion()
    assert tilefish.fusion(epilogues="library").canonical() == tilefish.fusion().canonical()  # a region it leaves is the same kernel
    lib = graph_of(lambda x, w, b, r: ((up(x, w).to(f32) + b.to(f32)) + r.to(f32)).to(bf16))
    assert _fusion.report(lib)["fusion"]["epilogues"] == "library"
    assert "epilogues" not in _fusion.report(graph_of(lambda x, w, b, r: ((up(x, w).to(f32) + b.to(f32))).to(bf16), "exact"))["fusion"]


def test_reference_mode_interprets_the_rewritten_program():
    """``mode="reference"`` of a rewritten graph is the formula on an f32 accumulator, rounded once (``R_acc(P)``), not the two roundings."""
    torch = pytest.importorskip("torch")
    g = graph_of(lambda x, w, b, r: relu((up(x, w).to(f32) + r.to(f32)) + b.to(f32)).to(bf16))
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(32, 256, generator=gen).to(torch.bfloat16)
    w = (torch.randn(512, 256, generator=gen) / 16).to(torch.bfloat16)
    b = torch.randn(512, generator=gen).to(torch.bfloat16)
    r = torch.randn(32, 512, generator=gen).to(torch.bfloat16)
    got = tilefish.run(g, x, w, b, r, mode="reference")
    want = torch.relu((x.float() @ w.float().t() + r.float()) + b.float()).to(torch.bfloat16)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_a_run_hands_the_c_of_a_matmul_step_to_the_backend():
    import test_planrun as tp

    class Backend(tp.FakeBackend):
        def gemm(self, spec, algo, a, b, bias, out, workspace, stream, index, c=None):
            self.log.append(("gemm", spec.stem, a, b, bias, c, out, workspace, stream))

    g = graph_of(lambda x, w, b, r: relu((up(x, w).to(f32) + r.to(f32)) + b.to(f32)).to(bf16))
    p = tilefish.plan(g)
    be = Backend()
    run = tp.PlanRunner(p, tp.FakeProvider(), be)
    x, w, b, r = (tp.FakeTensor(s, "bf16", (k + 1) << 32) for k, s in enumerate([(32, 256), (512, 256), (512,), (32, 512)]))
    out = run.run((x, w, b, r), {})
    (call,) = [e for e in be.log if e[0] == "gemm"]
    assert call[1:7] == ("up__relu_bias_c.M32", x._ptr, w._ptr, b._ptr, r._ptr, out._ptr)
