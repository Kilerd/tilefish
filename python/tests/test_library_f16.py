"""A pinned cuBLASLt GEMM of f16 operands, an f32 result, the layout ``NN``: declared, specified, written to a manifest and read back, with no GPU.

``@tilefish.library(..., impl="cublaslt.gemm" | "cublaslt.gemm_bias", layout="TN" | "NN")`` over ``a: Tensor[M, K, T]``, ``b: Tensor[N, K, T]``
(``TN``) or ``Tensor[K, N, T]`` (``NN``), ``bias: Tensor[N, D]`` and a result ``Tensor[M, N, D]``, with ``T`` bf16 or f16 and ``D`` ``T`` or f32. The
types and the layout reach the entry as fields it always had (``dtypes``, ``transa``, ``lda``, the operands' dtypes and shapes), and the layout
reaches the digest of the declaration; the opt-in to split-K partial sums in the output type is named by the operands' type
(``allow_f16_reduction``). The binding poses the problem with the types and the transposes of the form (``_Problem``, recorded here), and holds an
algorithm to the reduction rule in the words of its form. What it does on a GPU is ``test_gpu_gemm_f16.py``.
"""

import json

import pytest
from test_cublaslt import RecordingLib

import tilefish
from tilefish import AlgoTable, Dim, Manifest, Tensor, TilefishDeclarationError, TilefishManifestError, TilefishRuntimeError, bf16, f16, f32
from tilefish import _cublaslt as L
from tilefish import _library
from tilefish._library import BuiltLibrary, build_library, declaration_digest, library_specs, validate_entry
from tilefish.aot import write_manifest

M = Dim("M", buckets=[32, 64])
VERSION = 120804
SMS = 30
ALGO = "0123456789abcdef" * 8
CONFIG = {
    "id": 66, "tile_id": 80, "splitk_num": 1, "reduction_scheme": 0, "cta_swizzling": 0, "custom_option": 0, "stages_id": 35,
    "inner_shape_id": 0, "cluster_shape_id": 3,
}  # fmt: skip


def declare(fn, name="g", impl=None, **kw):
    impl = impl or ("cublaslt.gemm_bias" if "bias" in fn.__code__.co_varnames[: fn.__code__.co_argcount] else "cublaslt.gemm")
    return tilefish.library(name, acc=f32, impl=impl, **kw)(fn)


def tn_f16(a: Tensor[M, 1024, f16], b: Tensor[3072, 1024, f16]) -> Tensor[M, 3072, f16]: ...


def nn_f16(a: Tensor[M, 1024, f16], b: Tensor[1024, 3072, f16]) -> Tensor[M, 3072, f16]: ...


def nn_f16_bias(a: Tensor[M, 1024, f16], b: Tensor[1024, 3072, f16], bias: Tensor[3072, f16]) -> Tensor[M, 3072, f16]: ...


def tn_f16_f32(a: Tensor[M, 1024, f16], b: Tensor[3072, 1024, f16], bias: Tensor[3072, f32]) -> Tensor[M, 3072, f32]: ...


def nn_bf16_f32(a: Tensor[M, 1024, bf16], b: Tensor[1024, 3072, bf16]) -> Tensor[M, 3072, f32]: ...


def square_tn(a: Tensor[M, 1024, f16], b: Tensor[1024, 1024, f16]) -> Tensor[M, 1024, f16]: ...


def built(spec, arch="sm_103", algo=ALGO, version=VERSION, sm_count=SMS):
    return BuiltLibrary(spec, arch, algo, version, sm_count)


# -- the declaration ------------------------------------------------------------------------------------------------------------------------


def test_f16_operands_an_f32_result_and_the_layout_nn_are_declared():
    for fn, layout, form in [
        (tn_f16, "TN", L.Form("f16", "f16", "TN")),
        (nn_f16, "NN", L.Form("f16", "f16", "NN")),
        (nn_f16_bias, "NN", L.Form("f16", "f16", "NN")),
        (tn_f16_f32, "TN", L.Form("f16", "f32", "TN")),
        (nn_bf16_f32, "NN", L.Form("bf16", "f32", "NN")),
    ]:
        op = declare(fn, layout=layout)
        assert op.layout == layout and _library.form_of(op) == form
        assert [s.key for s in library_specs(op)] == [_library.table_key(m, 3072, 1024, "bias" in op.signature.params[-1].name, layout) for m in (32, 64)]
    assert declare(tn_f16).layout == "TN"  # the default


def _mixed(a: Tensor[M, 1024, f16], b: Tensor[3072, 1024, bf16]) -> Tensor[M, 3072, f16]: ...


def _f32_in(a: Tensor[M, 1024, f32], b: Tensor[3072, 1024, f32]) -> Tensor[M, 3072, f32]: ...


def _f16_to_bf16(a: Tensor[M, 1024, f16], b: Tensor[3072, 1024, f16]) -> Tensor[M, 3072, bf16]: ...


def _f32_bias_f16_out(a: Tensor[M, 1024, f16], b: Tensor[3072, 1024, f16], bias: Tensor[3072, f32]) -> Tensor[M, 3072, f16]: ...


def _f16_bias_f32_out(a: Tensor[M, 1024, f16], b: Tensor[3072, 1024, f16], bias: Tensor[3072, f16]) -> Tensor[M, 3072, f32]: ...


def _nn_k_mismatch(a: Tensor[M, 1024, f16], b: Tensor[3072, 1024, f16]) -> Tensor[M, 1024, f16]: ...


def test_the_types_go_together_and_the_bias_has_the_type_of_the_result():
    with pytest.raises(TilefishDeclarationError, match="a and b must both be bf16 or both be f16 tensors"):
        declare(_mixed)
    with pytest.raises(TilefishDeclarationError, match="a and b must both be bf16 or both be f16 tensors"):
        declare(_f32_in)
    with pytest.raises(TilefishDeclarationError, match="the result of f16 operands must be a f16 or an f32 tensor"):
        declare(_f16_to_bf16)
    # cuBLASLt 12.8.4 answers CUBLAS_STATUS_INVALID_VALUE for these two (measured): a declaration cannot have them
    with pytest.raises(TilefishDeclarationError, match="bias must be a f16 tensor of rank 1, the type of the result"):
        declare(_f32_bias_f16_out)
    with pytest.raises(TilefishDeclarationError, match="bias must be a f32 tensor of rank 1, the type of the result"):
        declare(_f16_bias_f32_out)


def test_the_layout_says_where_k_is_in_b():
    # b [1024, 3072] is [K, N] for NN; read as TN it is [N, K] with K = 3072, which a [M, 1024] cannot meet
    with pytest.raises(TilefishDeclarationError, match=r"a is \[M, K\] and b is \[N, K\]: their last axes must be the same \(1024 vs 3072\)"):
        declare(nn_f16)
    with pytest.raises(TilefishDeclarationError, match=r"a is \[M, K\] and b is \[K, N\] \(layout NN\): the last axis of a and the first of b must be the same"):
        declare(_nn_k_mismatch, layout="NN")
    with pytest.raises(TilefishDeclarationError, match=r"layout must be one of \['TN', 'NN'\]"):
        declare(tn_f16, layout="NT")
    with pytest.raises(TilefishDeclarationError, match="layout must be one of"):
        declare(tn_f16, layout="tn")
    with pytest.raises(TilefishDeclarationError, match="need an impl"):
        tilefish.library("x", layout="NN")


def test_the_opt_in_is_the_one_of_the_operands_type():
    assert declare(tn_f16, allow_f16_reduction=True).allow_f16_reduction is True
    assert declare(tn_f16_f32, allow_f16_reduction=True).allow_f16_reduction is True  # named by the operands, as torch's flag is
    with pytest.raises(TilefishDeclarationError, match="allow_bf16_reduction is the opt-in of bf16 operands; a GEMM of f16 operands opts in with allow_f16_reduction=True"):
        declare(tn_f16, allow_bf16_reduction=True)
    with pytest.raises(TilefishDeclarationError, match="allow_f16_reduction is the opt-in of f16 operands; a GEMM of bf16 operands opts in with allow_bf16_reduction=True"):
        declare(nn_bf16_f32, layout="NN", allow_f16_reduction=True)
    for bad in (1, "yes", None):
        if bad is None:
            continue
        with pytest.raises(TilefishDeclarationError, match="allow_f16_reduction must be a bool"):
            declare(tn_f16, allow_f16_reduction=bad)


def test_a_workspace_of_64_mib_is_declared_per_op():
    op = declare(nn_f16_bias, layout="NN", workspace_bytes=64 << 20)
    assert op.workspace_bytes == 67_108_864
    assert all(e.entry(SMS)["workspace_bytes"] == 67_108_864 for e in library_specs(op))
    assert declare(nn_f16, layout="NN").workspace_bytes == 8_519_680  # the default is per op too


# -- the digest and the entry ---------------------------------------------------------------------------------------------------------------


def test_the_layout_and_the_types_reach_the_digest():
    tn = declare(square_tn)
    nn = declare(square_tn, layout="NN")  # b [1024, 1024] is [N, K] or [K, N]: the signature alone does not tell them apart
    assert _library._canonical(nn)["layout"] == "NN" and "layout" not in _library._canonical(tn)
    assert declaration_digest(tn) != declaration_digest(nn)
    f16_out = declaration_digest(declare(tn_f16))
    f32_out = declaration_digest(declare(tn_f16_f32))
    assert f16_out != f32_out
    assert _library._canonical(declare(tn_f16, allow_f16_reduction=True))["allow_f16_reduction"] is True
    assert declaration_digest(declare(tn_f16, allow_f16_reduction=False)) == f16_out


def test_the_entry_of_the_layout_nn_says_the_problem_as_cublaslt_is_called():
    spec = library_specs(declare(nn_f16_bias, layout="NN"))[1]
    entry = spec.entry(SMS)
    assert (entry["transa"], entry["transb"]) == ("N", "N")
    assert (entry["m"], entry["n"], entry["k"]) == (64, 3072, 1024)
    assert (entry["lda"], entry["ldb"], entry["ldc"], entry["ldd"]) == (3072, 1024, 3072, 3072)  # b [k, n] is n x k column-major: lda = n
    assert entry["dtypes"] == {"a": "f16", "b": "f16", "bias": "f16", "out": "f16"}
    assert [(p["name"], p["dtype"], p["shape"]) for p in entry["abi"]["params"]] == [
        ("a", "f16", ["M", 1024]), ("b", "f16", [1024, 3072]), ("bias", "f16", [3072]), ("out", "f16", ["M", 3072]),
    ]  # fmt: skip
    d = L.describe(spec.m, spec.n, spec.k, spec.bias, form=spec.form)
    assert (d["transa"], d["transb"], d["lda"], d["ldb"], d["ldd"]) == (entry["transa"], entry["transb"], entry["lda"], entry["ldb"], entry["ldd"])
    assert (d["a_type"], d["d_type"], d["bias_type"], d["a_operand"]) == ("CUDA_R_16F", "CUDA_R_16F", "CUDA_R_16F", "B[K, N]")
    validate_entry(built(spec).entry())


def test_the_entry_of_an_f32_result_has_an_f32_bias_and_output():
    spec = library_specs(declare(tn_f16_f32))[0]
    entry = spec.entry(SMS)
    assert entry["dtypes"] == {"a": "f16", "b": "f16", "bias": "f32", "out": "f32"}
    assert (entry["transa"], entry["lda"]) == ("T", 1024)
    assert [p["dtype"] for p in entry["abi"]["params"]] == ["f16", "f16", "f32", "f32"]
    d = L.describe(spec.m, spec.n, spec.k, spec.bias, form=spec.form)
    assert (d["a_type"], d["b_type"], d["c_type"], d["d_type"], d["bias_type"]) == ("CUDA_R_16F", "CUDA_R_16F", "CUDA_R_32F", "CUDA_R_32F", "CUDA_R_32F")
    validate_entry(built(spec).entry())
    assert built(library_specs(declare(tn_f16, allow_f16_reduction=True))[0]).entry()["allow_f16_reduction"] is True


def refuses(op, change, message, index=0):
    entry = built(library_specs(op)[index]).entry()
    change(entry)
    with pytest.raises(TilefishManifestError, match=message):
        validate_entry(entry)


def test_the_reader_holds_the_types_and_the_layout_together():
    nn = declare(nn_f16_bias, layout="NN")
    refuses(nn, lambda e: e.update(lda=1024), "the leading dimensions of the layout NN must be lda = n, ldb = k and ldc = ldd = n")
    refuses(nn, lambda e: e.update(transa="T"), "the leading dimensions must be lda = ldb = k")  # TN with lda = n
    refuses(nn, lambda e: e.update(transa="T", lda=1024), r"abi.params\[1\].shape\[0\] is 1024 = 1024, but m, n and k say 3072")  # b is [k, n]
    refuses(nn, lambda e: e.update(transa="X"), "transa and transb must be")
    refuses(nn, lambda e: e.update(transb="T"), "transa and transb must be")
    refuses(nn, lambda e: e["dtypes"].update(b="bf16"), "dtypes must be")
    refuses(nn, lambda e: e["dtypes"].update(bias="f32"), "dtypes must be")  # a bias of another type than the result
    refuses(nn, lambda e: e["dtypes"].update(out="bf16"), "dtypes must be")
    refuses(nn, lambda e: e["abi"]["params"][2].update(dtype="f32"), r"abi.params\[2\] must be the f16 pointer #2")
    refuses(nn, lambda e: e["abi"]["params"][0].update(dtype="bf16"), r"abi.params\[0\] must be the f16 pointer #0")
    f32_out = declare(tn_f16_f32)
    refuses(f32_out, lambda e: e["dtypes"].update(bias="f16"), "dtypes must be")
    refuses(f32_out, lambda e: e["abi"]["params"][3].update(dtype="f16"), r"abi.params\[3\] must be the f32 pointer #3")
    # an opt-in of the other type
    refuses(nn, lambda e: e.update(allow_bf16_reduction=True), "allow_bf16_reduction is the opt-in of bf16 operands, and the operands of this entry are f16")
    refuses(declare(nn_bf16_f32, layout="NN"), lambda e: e.update(allow_f16_reduction=True), "allow_f16_reduction is the opt-in of f16 operands")
    refuses(nn, lambda e: e.update(allow_f16_reduction=False), r"allow_f16_reduction is absent \(the default\) or true")


def test_a_manifest_holds_f16_and_nn_entries_and_a_replay_takes_them(tmp_path, monkeypatch):
    ops = [declare(nn_f16_bias, "proj", layout="NN", workspace_bytes=64 << 20), declare(tn_f16_f32, "up", allow_f16_reduction=True)]
    library = [built(s) for op in ops for s in library_specs(op)]
    path = tmp_path / "m" / "manifest.json"
    write_manifest([], path, cubins=True, library=library)
    m = Manifest.load(tmp_path / "m")
    assert sorted(e["stem"] for e in m.library_entries) == ["proj.M32", "proj.M64", "up.M32", "up.M64"]
    monkeypatch.setattr(L, "version", lambda: VERSION)
    for op in ops:
        for spec in library_specs(op):
            got = _library.built_from_manifest(m, spec, "sm_103", SMS)
            assert got.algo == ALGO and got.spec is spec


# -- imported algorithms --------------------------------------------------------------------------------------------------------------------


def test_a_table_row_of_another_layout_says_it():
    rows = {(m, 3072, 1024, True, "NN"): CONFIG for m in (32, 64)}
    t = AlgoTable("sm_103", SMS, VERSION, rows)
    assert set(t.algos) == {(32, 3072, 1024, True, "NN"), (64, 3072, 1024, True, "NN")}
    op = declare(nn_f16_bias, layout="NN", algos=t)
    assert [s.key for s in library_specs(op)] == [(32, 3072, 1024, True, "NN"), (64, 3072, 1024, True, "NN")]
    # a table of TN rows does not cover the NN declaration of the same sizes
    tn_rows = AlgoTable("sm_103", SMS, VERSION, {(m, 3072, 1024, True): CONFIG for m in (32, 64)})
    with pytest.raises(TilefishDeclarationError, match=r"has no algorithm for \(m, n, k, bias, layout\) = \(32, 3072, 1024, True, 'NN'\)"):
        declare(nn_f16_bias, layout="NN", algos=tn_rows)
    # (m, n, k, bias, "TN") is (m, n, k, bias); a table may not have both
    assert AlgoTable("sm_103", SMS, VERSION, {(32, 3072, 1024, True, "TN"): CONFIG}).algos == {(32, 3072, 1024, True): CONFIG}
    with pytest.raises(TilefishDeclarationError, match="two rows are for"):
        AlgoTable("sm_103", SMS, VERSION, {(32, 3072, 1024, True, "TN"): CONFIG, (32, 3072, 1024, True): CONFIG})
    for bad in [(32, 3072, 1024, True, "NT"), (32, 3072, 1024, True, "nn"), (32, 3072, 1024, True, "NN", 1)]:
        with pytest.raises(TilefishDeclarationError, match="a key is"):
            AlgoTable("sm_103", SMS, VERSION, {bad: CONFIG})
    # the rows of the declaration reach its digest, without the layout (the declaration says it)
    canonical = _library._canonical(op)
    assert canonical["algos"][0]["algos"] == [[32, 3072, 1024, True, CONFIG], [64, 3072, 1024, True, CONFIG]]


def test_an_imported_configuration_is_encoded_with_the_types_of_the_declaration(monkeypatch):
    calls = []
    monkeypatch.setattr(_library, "_device_index", lambda i: 0)
    monkeypatch.setattr(_library, "_capability", lambda i: 103)
    monkeypatch.setattr(_library, "sm_count", lambda i: SMS)
    monkeypatch.setattr(L, "version", lambda: VERSION)
    monkeypatch.setattr(L, "encode_algo", lambda cfg, **kw: calls.append(("encode", cfg, kw)) or ALGO)
    monkeypatch.setattr(L, "check_algo", lambda *a, **kw: calls.append(("check", a, kw)) or 0)
    t = AlgoTable("sm_103", SMS, VERSION, {(m, 3072, 1024, False): CONFIG for m in (32, 64)})
    spec = library_specs(declare(tn_f16, algos=t, allow_f16_reduction=True))[0]
    got = build_library(spec, arch="sm_103")
    assert got.algo == ALGO and got.entry()["algo_source"] == "imported" and got.entry()["allow_f16_reduction"] is True
    form = L.Form("f16", "f16", "TN")
    opt_in = {"allow_bf16_reduction": False, "allow_f16_reduction": True}
    assert calls == [
        ("encode", CONFIG, {"device_index": 0, "form": form}),
        ("check", (32, 3072, 1024, False, ALGO), {"workspace_bytes": 8_519_680, "device_index": 0, "form": form, "sm_count_target": None, **opt_in}),
    ]


# -- the binding: the problem of a form, the rule in its words --------------------------------------------------------------------------------


def test_the_problem_of_f16_nn_with_a_bias_has_f16_layouts_and_bias_and_no_transpose():
    lib = RecordingLib()
    with L._Problem(lib, 64, 3072, 1024, True, 0x1000, None, L.Form("f16", "f16", "NN")):
        pass
    assert [a[1:] for a in lib.named("cublasLtMatrixLayoutCreate")] == [
        [2, 3072, 1024, 3072],  # the A operand: tilefish's B[1024, 3072] is 3072 x 1024 column-major, lda = 3072, used as it is
        [2, 1024, 64, 1024],  # the B operand: tilefish's A[64, 1024]
        [2, 3072, 64, 3072],  # D (and C)
    ]
    attrs = {a[1]: a[2] for a in lib.named("cublasLtMatmulDescSetAttribute")}
    assert attrs == {3: 0, 4: 0, 7: 4, 26: 2, 8: 0x1000}  # TRANSA = N, TRANSB = N, BIAS, BIAS_DATA_TYPE = CUDA_R_16F


def test_the_problem_of_an_f32_result_has_f32_c_d_and_bias():
    lib = RecordingLib()
    with L._Problem(lib, 8, 16, 32, True, 0x1000, None, L.Form("f16", "f32", "TN")):
        pass
    assert [a[1:] for a in lib.named("cublasLtMatrixLayoutCreate")] == [[2, 32, 16, 32], [2, 32, 8, 32], [0, 16, 8, 16]]
    assert {a[1]: a[2] for a in lib.named("cublasLtMatmulDescSetAttribute")}[26] == 0  # CUDA_R_32F


def test_the_form_and_the_opt_in_are_checked_before_the_library(monkeypatch):
    monkeypatch.setattr(L, "_lib", lambda: (_ for _ in ()).throw(AssertionError("the library was reached")))
    with pytest.raises(TilefishRuntimeError, match=r"the operands are one of \['bf16', 'f16'\]"):
        L.Form("f32", "f32")
    with pytest.raises(TilefishRuntimeError, match="the result of f16 operands is f16 or f32"):
        L.Form("f16", "bf16")
    with pytest.raises(TilefishRuntimeError, match=r"the layout is one of \['TN', 'NN'\]"):
        L.Form("f16", "f16", "TT")
    with pytest.raises(TilefishRuntimeError, match="form must be a tilefish._cublaslt.Form"):
        L.check_algo(8, 16, 32, False, ALGO, workspace_bytes=0, form=("f16", "f16", "NN"))
    with pytest.raises(TilefishRuntimeError, match="allow_bf16_reduction is the opt-in of bf16 operands; f16 operands opt in with allow_f16_reduction=True"):
        L.pin_algo(8, 16, 32, form=L.Form("f16", "f16"), allow_bf16_reduction=True)
    with pytest.raises(TilefishRuntimeError, match="allow_f16_reduction must be a bool"):
        L.matmul(16, 32, 48, None, 8, 16, 32, ALGO, workspace_bytes=0, stream=0, allow_f16_reduction=1)


def test_the_reduction_rule_speaks_of_the_output_type_and_the_opt_in_of_the_operands():
    f16 = L.Form("f16", "f16", "NN")
    assert L.reduction_problem(4, 4, False, False, f16) == (
        "reduction scheme 4 adds the split-K partial sums in f16, which rounds more than once (only no split-K, 0, and f32 partial sums, 2, keep the "
        "one rounding); a declaration may opt in with allow_f16_reduction=True"
    )
    assert "a sum already rounded to f16" in L.reduction_problem(4, 2, False, True, f16)
    assert "opts in with allow_f16_reduction=True" in L.reduction_problem(4, 2, False, True, f16)
    f32_out = L.Form("f16", "f32")
    assert L.reduction_problem(4, 1, False, False, f32_out).startswith("reduction scheme 1 adds the split-K partial sums in the output (f32)")
    for form in (f16, f32_out):
        assert L.reduction_problem(4, 4, True, True, form) is None and L.reduction_problem(4, 2, False, False, form) is None
    # the default form keeps the words of 0.0.5
    assert L.reduction_problem(4, 4, False, False) == L.reduction_problem(4, 4, False, False, L.Form())
    assert "adds the split-K partial sums in bf16" in L.reduction_problem(4, 4, False, False)


def test_describe_of_each_form():
    d = L.describe(64, 3072, 1024, False, form=L.Form("f16", "f16", "NN"), allow_f16_reduction=True)
    assert (d["dtype"], d["out_dtype"], d["layout"], d["transa"], d["transb"]) == ("f16", "f16", "NN", "N", "N")
    assert (d["lda"], d["ldb"], d["ldc"], d["ldd"]) == (3072, 1024, 3072, 3072)
    assert d["reduction_schemes"] == [0, 2, 1, 4]
    assert L.describe(64, 3072, 1024, False)["out_dtype"] == "bf16"


def test_a_rust_launcher_takes_the_tensors_of_the_types_of_the_entry(tmp_path):
    from tilefish.hosts import rust

    ops = [declare(nn_f16_bias, "proj", layout="NN"), declare(tn_f16_f32, "up", allow_f16_reduction=True)]
    path = tmp_path / "m" / "manifest.json"
    write_manifest([], path, cubins=True, library=[built(s) for op in ops for s in library_specs(op)])
    m = Manifest.load(tmp_path / "m")
    source = rust.render(m.check_pin(m.digest), ops=ops)
    assert "b: TensorRef<'a, F16>" in source and "out: TensorMut<'a, F32>" in source and "bias: TensorRef<'a, F32>" in source
    assert "`out[32, 3072] = a[32, 1024] @ b[1024, 3072] + bias[3072]` on f16 operands" in source and "/// f16 once" in source
    assert "`out[32, 3072] = a[32, 1024] @ b[3072, 1024]^T + bias[3072]` on f16 operands" in source and "/// f32 once" in source
    assert "split-K to add its partial sums in the output type (`allow_f16_reduction`)" in source


def test_the_committed_library_vectors_are_what_python_decides_now():
    """``crates/tilefish/tests/data/python_library.json`` (the entries of every form and their changes, with Python's verdict) is what the strict
    reader decides now; ``tests/python_library.rs`` holds the Rust reader to it."""
    from conftest import REPO_ROOT
    from export_rust_vectors import library_entries

    committed = json.loads((REPO_ROOT / "crates" / "tilefish" / "tests" / "data" / "python_library.json").read_text())
    made = json.loads(json.dumps(library_entries()))
    assert committed == made
    assert sum(c["accepted"] for c in made) >= 10 and sum(not c["accepted"] for c in made) >= 70
