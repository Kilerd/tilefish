"""The GEMM layouts ``TT`` and ``NT`` (``a`` stored ``[K, M]``): declared, specified, written to a manifest and read back, with no GPU.

``@tilefish.library(..., layout="TT" | "NT")`` over ``a: Tensor[K, M, T]`` with the ``b`` of ``TN`` (``[N, K]``) or of ``NN`` (``[K, N]``), for bf16,
f16 and f32 operands: ``out = a^T @ b^T`` and ``a^T @ b``. The layout reaches the entry as fields it always had (``transa``, ``transb``, ``lda``,
``ldb``, the shape of ``a``) and the digest of the declaration; the binding poses the problem with ``transb = T`` and ``ldb = m`` (``_Problem``,
recorded here); a table row says the layout in its key; the reference is torch's matmul of the transposed views. What it does on a GPU is
``test_gpu_gemm_layouts.py``.
"""

import pytest
from test_cublaslt import RecordingLib

import tilefish
from tilefish import AlgoTable, Dim, Manifest, Tensor, TilefishDeclarationError, TilefishManifestError, bf16, f16, f32
from tilefish import _cublaslt as L
from tilefish import _library
from tilefish._library import BuiltLibrary, declaration_digest, library_specs, validate_entry
from tilefish.aot import write_manifest

M = Dim("M", buckets=[32, 64])
VERSION = 120804
SMS = 30
ALGO = "0123456789abcdef" * 8
CONFIG = {
    "id": 66, "tile_id": 80, "splitk_num": 1, "reduction_scheme": 0, "cta_swizzling": 0, "custom_option": 0, "stages_id": 35,
    "inner_shape_id": 0, "cluster_shape_id": 3,
}  # fmt: skip


def declare(fn, name="g", **kw):
    impl = "cublaslt.gemm_bias" if "bias" in fn.__code__.co_varnames[: fn.__code__.co_argcount] else "cublaslt.gemm"
    return tilefish.library(name, acc=f32, impl=impl, **kw)(fn)


def built(spec, sm_count=SMS):
    return BuiltLibrary(spec, "sm_103", ALGO, VERSION, sm_count)


def tt_bf16(a: Tensor[1024, M, bf16], b: Tensor[3072, 1024, bf16]) -> Tensor[M, 3072, bf16]: ...


def nt_f16_bias(a: Tensor[1024, M, f16], b: Tensor[1024, 3072, f16], bias: Tensor[3072, f16]) -> Tensor[M, 3072, f16]: ...


def tt_f16_f32(a: Tensor[1024, M, f16], b: Tensor[3072, 1024, f16], bias: Tensor[3072, f32]) -> Tensor[M, 3072, f32]: ...


def nt_f32(a: Tensor[8, M, f32], b: Tensor[8, 4096, f32]) -> Tensor[M, 4096, f32]: ...


def tt_f32_k8(a: Tensor[8, M, f32], b: Tensor[1024, 8, f32]) -> Tensor[M, 1024, f32]: ...


# -- the declaration ------------------------------------------------------------------------------------------------------------------------


def test_a_stored_k_by_m_is_declared_for_every_operand_type():
    for fn, layout, form, n, k in [
        (tt_bf16, "TT", L.Form("bf16", "bf16", "TT"), 3072, 1024),
        (nt_f16_bias, "NT", L.Form("f16", "f16", "NT"), 3072, 1024),
        (tt_f16_f32, "TT", L.Form("f16", "f32", "TT"), 3072, 1024),
        (nt_f32, "NT", L.Form("f32", "f32", "NT"), 4096, 8),
        (tt_f32_k8, "TT", L.Form("f32", "f32", "TT"), 1024, 8),
    ]:
        op = declare(fn, layout=layout)
        assert op.layout == layout and _library.form_of(op) == form
        bias = "bias" in op.signature.params[-1].name
        assert [s.key for s in library_specs(op)] == [(m, n, k, bias, layout) for m in (32, 64)]
        assert [(s.m, s.n, s.k) for s in library_specs(op)] == [(32, n, k), (64, n, k)]


def _tt_k_mismatch(a: Tensor[1024, M, f16], b: Tensor[1024, 3072, f16]) -> Tensor[M, 1024, f16]: ...


def _nt_k_mismatch(a: Tensor[1024, M, f16], b: Tensor[3072, 1024, f16]) -> Tensor[M, 1024, f16]: ...


def _one_operand(a: Tensor[1024, M, f16]) -> Tensor[M, 1024, f16]: ...


def test_the_layout_says_where_k_is_in_a_and_in_b():
    # b [1024, 3072] read as TT is [N, K] with K = 3072, which an a [1024, M] (K = 1024) cannot meet
    with pytest.raises(TilefishDeclarationError, match=r"a is \[K, M\] and b is \[N, K\] \(layout TT\): the first axis of a and the last of b must be the same \(1024 vs 3072\)"):
        declare(_tt_k_mismatch, layout="TT")
    with pytest.raises(TilefishDeclarationError, match=r"a is \[K, M\] and b is \[K, N\] \(layout NT\): their first axes must be the same \(1024 vs 3072\)"):
        declare(_nt_k_mismatch, layout="NT")
    # a [K, M] declared as TN: its M is 1024 and its K the dimension M, which b [3072, 1024] cannot meet
    with pytest.raises(TilefishDeclarationError, match=r"their last axes must be the same \(M vs 1024\)"):
        declare(tt_bf16)
    with pytest.raises(TilefishDeclarationError, match=r"takes 2 input tensors \(a: Tensor\[K, M, T\], b: Tensor\[K, N, T\]\)"):
        declare(_one_operand, layout="NT")
    with pytest.raises(TilefishDeclarationError, match=r"layout must be one of \['TN', 'NN', 'TT', 'NT'\].*TT and NT: a is \[K, M\]"):
        declare(tt_bf16, layout="tt")


def test_the_layout_reaches_the_digest_of_the_declaration():
    def square(a: Tensor[1024, 1024, f16], b: Tensor[1024, 1024, f16]) -> Tensor[1024, 1024, f16]: ...

    digests = {layout: declaration_digest(declare(square, layout=layout)) for layout in ("TN", "NN", "TT", "NT")}
    assert len(set(digests.values())) == 4  # a square a and b do not tell the layouts apart: the canonical declaration does
    assert _library._canonical(declare(square, layout="TT"))["layout"] == "TT" and "layout" not in _library._canonical(declare(square))


# -- the entry ------------------------------------------------------------------------------------------------------------------------------


def test_the_entry_of_a_stored_k_by_m_says_the_problem_as_cublaslt_is_called():
    spec = library_specs(declare(nt_f16_bias, layout="NT"))[1]
    entry = spec.entry(SMS)
    assert (entry["transa"], entry["transb"]) == ("N", "T")
    assert (entry["m"], entry["n"], entry["k"]) == (64, 3072, 1024)
    assert (entry["lda"], entry["ldb"], entry["ldc"], entry["ldd"]) == (3072, 64, 3072, 3072)  # a [k, m] is m x k column-major: ldb = m
    assert [(p["name"], p["shape"]) for p in entry["abi"]["params"]] == [("a", [1024, "M"]), ("b", [1024, 3072]), ("bias", [3072]), ("out", ["M", 3072])]
    d = L.describe(spec.m, spec.n, spec.k, spec.bias, form=spec.form)
    assert (d["transa"], d["transb"], d["lda"], d["ldb"], d["ldd"]) == (entry["transa"], entry["transb"], entry["lda"], entry["ldb"], entry["ldd"])
    assert (d["a_operand"], d["b_operand"]) == ("B[K, N]", "A[K, M]")
    validate_entry(built(spec).entry())
    tt = library_specs(declare(tt_f32_k8, layout="TT"))[0].entry(SMS)
    assert (tt["transa"], tt["transb"], tt["lda"], tt["ldb"]) == ("T", "T", 8, 32)
    assert tt["dtypes"] == {"a": "f32", "b": "f32", "out": "f32"}


def refuses(op, change, message, index=0):
    entry = built(library_specs(op)[index]).entry()
    change(entry)
    with pytest.raises(TilefishManifestError, match=message):
        validate_entry(entry)


def test_the_reader_holds_the_layout_to_the_leading_dimensions_and_the_shapes():
    tt = declare(tt_bf16, layout="TT")
    refuses(tt, lambda e: e.update(ldb=1024), r"the leading dimensions of the layout TT must be lda = k, ldb = m and ldc = ldd = n \(cuBLAS sees b as its A, a \[k, m\] as its B")
    refuses(tt, lambda e: e.update(transb="N"), "the leading dimensions must be lda = ldb = k")  # TN with ldb = m
    refuses(tt, lambda e: e.update(transb="N", ldb=1024), r"abi.params\[0\].shape\[0\] is 1024 = 1024, but m, n and k say 32")  # a is [k, m]
    refuses(tt, lambda e: e.update(transa="N"), r"the leading dimensions of the layout NT must be lda = n, ldb = m")
    refuses(tt, lambda e: e.update(transb="t"), "transa and transb must each be 'T' or 'N'")
    nt = declare(nt_f16_bias, layout="NT")
    refuses(nt, lambda e: e.update(lda=1024), r"the leading dimensions of the layout NT must be lda = n, ldb = m and ldc = ldd = n \(cuBLAS sees b \[k, n\] as its A, a \[k, m\] as its B")
    refuses(nt, lambda e: e["abi"]["params"][0].update(shape=["M", 1024]), "but m, n and k say")
    validate_entry(built(library_specs(nt)[0]).entry())


def test_a_manifest_holds_entries_of_every_layout_and_a_replay_takes_them(tmp_path, monkeypatch):
    ops = [declare(tt_bf16, "tt", layout="TT", heuristic="torch", allow_bf16_reduction=True), declare(nt_f32, "nt", layout="NT")]
    library = [built(s) for op in ops for s in library_specs(op)]
    write_manifest([], tmp_path / "m" / "manifest.json", cubins=True, library=library)
    m = Manifest.load(tmp_path / "m")
    assert sorted((e["stem"], e["transa"] + e["transb"]) for e in m.library_entries) == [
        ("nt.M32", "NT"), ("nt.M64", "NT"), ("tt.M32", "TT"), ("tt.M64", "TT"),
    ]  # fmt: skip
    monkeypatch.setattr(L, "version", lambda: VERSION)
    for op in ops:
        for spec in library_specs(op):
            assert _library.built_from_manifest(m, spec, "sm_103", SMS).algo == ALGO


# -- imported algorithms --------------------------------------------------------------------------------------------------------------------


def test_a_table_row_of_a_stored_k_by_m_says_its_layout():
    rows = {(m, 4096, 8, False, "NT"): CONFIG for m in (32, 64)}
    op = declare(nt_f32, layout="NT", algos=AlgoTable("sm_103", SMS, VERSION, rows))
    assert [s.key for s in library_specs(op)] == [(32, 4096, 8, False, "NT"), (64, 4096, 8, False, "NT")]
    with pytest.raises(TilefishDeclarationError, match=r"has no algorithm for \(m, n, k, bias, layout\) = \(32, 4096, 8, False, 'NT'\)"):
        declare(nt_f32, layout="NT", algos=AlgoTable("sm_103", SMS, VERSION, {(m, 4096, 8, False, "NN"): CONFIG for m in (32, 64)}))


# -- the binding ----------------------------------------------------------------------------------------------------------------------------


def test_the_problem_of_a_stored_k_by_m_has_transb_t_and_ldb_m():
    lib = RecordingLib()
    with L._Problem(lib, 64, 3072, 1024, False, 0x1000, None, L.Form("bf16", "bf16", "TT")):
        pass
    assert [a[1:] for a in lib.named("cublasLtMatrixLayoutCreate")] == [
        [14, 1024, 3072, 1024],  # the A operand: tilefish's B[3072, 1024] is 1024 x 3072 column-major, used transposed
        [14, 64, 1024, 64],  # the B operand: tilefish's A[1024, 64] is 64 x 1024 column-major, ldb = 64, used transposed
        [14, 3072, 64, 3072],  # D (and C)
    ]
    assert {a[1]: a[2] for a in lib.named("cublasLtMatmulDescSetAttribute")} == {3: 1, 4: 1, 7: 1}  # TRANSA = T, TRANSB = T, no epilogue
    lib = RecordingLib()
    with L._Problem(lib, 8, 16, 32, True, 0x1000, None, L.Form("f32", "f32", "NT")):
        pass
    assert [a[1:] for a in lib.named("cublasLtMatrixLayoutCreate")] == [[0, 16, 32, 16], [0, 8, 32, 8], [0, 16, 8, 16]]
    assert {a[1]: a[2] for a in lib.named("cublasLtMatmulDescSetAttribute")}[4] == 1  # TRANSB = T


def test_the_reference_multiplies_the_transposed_views():
    torch = pytest.importorskip("torch")
    a, b = torch.arange(6.0).reshape(3, 2), torch.arange(12.0).reshape(4, 3)  # a [k=3, m=2], b [n=4, k=3]
    tt = _library.reference_gemm({"a": a, "b": b}, "TT", "f32")
    assert torch.equal(tt, a.t() @ b.t())
    nt = _library.reference_gemm({"a": a, "b": b.t().contiguous()}, "NT", "f32")
    assert torch.equal(nt, a.t() @ b.t())


def test_a_rust_launcher_documents_the_transposed_a(tmp_path):
    from tilefish.hosts import rust

    ops = [declare(tt_bf16, "tt", layout="TT"), declare(nt_f16_bias, "nt", layout="NT")]
    write_manifest([], tmp_path / "m" / "manifest.json", cubins=True, library=[built(s) for op in ops for s in library_specs(op)])
    m = Manifest.load(tmp_path / "m")
    source = rust.render(m.check_pin(m.digest), ops=ops)
    assert "`out[32, 3072] = a[1024, 32]^T @ b[3072, 1024]^T` on bf16 operands" in source
    assert "`out[64, 3072] = a[1024, 64]^T @ b[1024, 3072] + bias[3072]` on f16 operands" in source
    assert "Columns of `a` (`[K, M]`: the layout TT), and rows of the result." in source
    assert "The rows of `a` and the columns of `b`." in source and "The rows of `a` and of `b`." in source
