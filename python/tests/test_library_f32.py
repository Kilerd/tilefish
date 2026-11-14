"""A pinned cuBLASLt GEMM of f32 operands and an f32 result, TF32 off: declared, specified, written to a manifest and read back, with no GPU.

``@tilefish.library(..., impl="cublaslt.gemm" | "cublaslt.gemm_bias", layout="TN" | "NN")`` over ``a: Tensor[M, K, f32]``, ``b: Tensor[N, K, f32]``
(``TN``) or ``Tensor[K, N, f32]`` (``NN``), ``bias: Tensor[N, f32]`` and a result ``Tensor[M, N, f32]``: no 16-bit result of f32 operands. The
compute type is ``CUBLAS_COMPUTE_32F`` (never ``_FAST_TF32`` nor ``_EMULATED_16BFX9``), and the vetting of an algorithm holds it to that: cuBLASLt
must encode ``CUBLAS_COMPUTE_32F`` as itself (an environment with ``NVIDIA_TF32_OVERRIDE=1`` or ``CUBLAS_EMULATE_SINGLE_PRECISION=1`` makes it
encode, and run, TF32 or a bf16x9 emulation), and the numerical-implementation flags, where the library reports them, say f32 inputs. The opt-in
to split-K partial sums in the output is ``allow_f32_reduction``. Entries of tilefish 0.0.5 and of the f16 forms keep their bytes and digests.
What it does on a GPU is ``test_gpu_gemm_f32.py``.
"""

import hashlib
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
VERSION = 130100
SMS = 30
ALGO = "0123456789abcdef" * 8
CONFIG = {
    "id": 76, "tile_id": 18, "splitk_num": 1, "reduction_scheme": 0, "cta_swizzling": 0, "custom_option": 0, "stages_id": 3,
    "inner_shape_id": 0, "cluster_shape_id": 2,
}  # fmt: skip
F32 = L.Form("f32", "f32", "TN")
F32_NN = L.Form("f32", "f32", "NN")


def declare(fn, name="g", impl=None, **kw):
    impl = impl or ("cublaslt.gemm_bias" if "bias" in fn.__code__.co_varnames[: fn.__code__.co_argcount] else "cublaslt.gemm")
    return tilefish.library(name, acc=f32, impl=impl, **kw)(fn)


def tn(a: Tensor[M, 1024, f32], b: Tensor[3072, 1024, f32]) -> Tensor[M, 3072, f32]: ...


def tn_bias(a: Tensor[M, 1024, f32], b: Tensor[3072, 1024, f32], bias: Tensor[3072, f32]) -> Tensor[M, 3072, f32]: ...


def nn(a: Tensor[M, 1024, f32], b: Tensor[1024, 3072, f32]) -> Tensor[M, 3072, f32]: ...


def nn_bias(a: Tensor[M, 1024, f32], b: Tensor[1024, 3072, f32], bias: Tensor[3072, f32]) -> Tensor[M, 3072, f32]: ...


def built(spec, arch="sm_103", algo=ALGO, version=VERSION, sm_count=SMS):
    return BuiltLibrary(spec, arch, algo, version, sm_count)


# -- the declaration ------------------------------------------------------------------------------------------------------------------------


def test_f32_operands_and_an_f32_result_are_declared_in_both_layouts_with_and_without_a_bias():
    for fn, layout in [(tn, "TN"), (tn_bias, "TN"), (nn, "NN"), (nn_bias, "NN")]:
        op = declare(fn, layout=layout)
        assert op.layout == layout and _library.form_of(op) == L.Form("f32", "f32", layout)
        bias = "bias" in op.signature.params[-1].name
        assert [s.key for s in library_specs(op)] == [_library.table_key(m, 3072, 1024, bias, layout) for m in (32, 64)]
        assert op.allow_f32_reduction is False
    assert declare(tn, allow_f32_reduction=True).allow_f32_reduction is True


def _to_bf16(a: Tensor[M, 1024, f32], b: Tensor[3072, 1024, f32]) -> Tensor[M, 3072, bf16]: ...


def _to_f16(a: Tensor[M, 1024, f32], b: Tensor[3072, 1024, f32]) -> Tensor[M, 3072, f16]: ...


def _f16_bias(a: Tensor[M, 1024, f32], b: Tensor[3072, 1024, f32], bias: Tensor[3072, f16]) -> Tensor[M, 3072, f32]: ...


def _f32_f16(a: Tensor[M, 1024, f32], b: Tensor[3072, 1024, f16]) -> Tensor[M, 3072, f32]: ...


def test_f32_operands_have_an_f32_result_and_bias_and_both_operands_are_f32():
    for fn in (_to_bf16, _to_f16):
        with pytest.raises(TilefishDeclarationError, match=r"the result of f32 operands must be an f32 tensor \(they are multiplied as f32, TF32 off"):
            declare(fn)
    with pytest.raises(TilefishDeclarationError, match="bias must be a f32 tensor of rank 1, the type of the result"):
        declare(_f16_bias)
    with pytest.raises(TilefishDeclarationError, match="a and b must both be bf16, both be f16 or both be f32 tensors"):
        declare(_f32_f16)
    with pytest.raises(TilefishRuntimeError, match="the result of f32 operands is f32, got 'bf16'"):
        L.Form("f32", "bf16")
    with pytest.raises(TilefishRuntimeError, match="the result of f32 operands is f32, got 'f16'"):
        L.Form("f32", "f16")


def test_the_opt_in_of_f32_operands_is_allow_f32_reduction():
    with pytest.raises(TilefishDeclarationError, match="allow_f16_reduction is the opt-in of f16 operands; a GEMM of f32 operands opts in with allow_f32_reduction=True"):
        declare(tn, allow_f16_reduction=True)
    with pytest.raises(TilefishDeclarationError, match="allow_bf16_reduction is the opt-in of bf16 operands; a GEMM of f32 operands opts in with allow_f32_reduction=True"):
        declare(tn, allow_bf16_reduction=True)

    def bf16_op(a: Tensor[M, 1024, bf16], b: Tensor[3072, 1024, bf16]) -> Tensor[M, 3072, f32]: ...

    with pytest.raises(TilefishDeclarationError, match="allow_f32_reduction is the opt-in of f32 operands; a GEMM of bf16 operands opts in with allow_bf16_reduction=True"):
        declare(bf16_op, allow_f32_reduction=True)
    for bad in (1, "yes", 0.0):
        with pytest.raises(TilefishDeclarationError, match="allow_f32_reduction must be a bool"):
            declare(tn, allow_f32_reduction=bad)
    with pytest.raises(TilefishDeclarationError, match="allow_f32_reduction, algos, layout and sm_count_target need an impl"):
        tilefish.library("x", allow_f32_reduction=True)
    # the binding: the same rule, before the library is reached
    with pytest.raises(TilefishRuntimeError, match="allow_f16_reduction is the opt-in of f16 operands; f32 operands opt in with allow_f32_reduction=True"):
        L.describe(8, 16, 32, False, form=F32, allow_f16_reduction=True)
    with pytest.raises(TilefishRuntimeError, match="allow_f32_reduction is the opt-in of f32 operands; bf16 operands opt in with allow_bf16_reduction=True"):
        L.describe(8, 16, 32, False, allow_f32_reduction=True)
    with pytest.raises(TilefishRuntimeError, match="allow_f32_reduction must be a bool"):
        L.describe(8, 16, 32, False, form=F32, allow_f32_reduction=1)


# -- the problem ------------------------------------------------------------------------------------------------------------------------------


def test_describe_of_the_f32_problem():
    d = L.describe(64, 3072, 1024, True, form=F32_NN, allow_f32_reduction=True)
    assert (d["dtype"], d["out_dtype"], d["layout"]) == ("f32", "f32", "NN")
    assert (d["a_type"], d["b_type"], d["c_type"], d["d_type"], d["bias_type"]) == ("CUDA_R_32F",) * 5
    assert (d["compute_type"], d["scale_type"]) == ("CUBLAS_COMPUTE_32F", "CUDA_R_32F")  # never CUBLAS_COMPUTE_32F_FAST_TF32
    assert (d["transa"], d["transb"], d["lda"], d["ldb"], d["ldc"], d["ldd"]) == ("N", "N", 3072, 1024, 3072, 3072)
    assert d["reduction_schemes"] == [0, 2, 1, 4]
    plain = L.describe(64, 3072, 1024, True, form=F32)
    assert (plain["transa"], plain["lda"], plain["reduction_schemes"]) == ("T", 1024, [0])
    assert L.describe(64, 3072, 1024, False, form=F32)["reduction_schemes"] == [0, 2]
    assert (F32.what, F32.label, F32.opt_in) == ("f32", " (f32, TN)", "allow_f32_reduction")


def test_the_problem_of_f32_operands_has_f32_layouts_and_bias():
    lib = RecordingLib()
    with L._Problem(lib, 8, 16, 32, True, 0x1000, None, F32_NN):
        pass
    assert [a[1:] for a in lib.named("cublasLtMatrixLayoutCreate")] == [[0, 16, 32, 16], [0, 32, 8, 32], [0, 16, 8, 16]]  # CUDA_R_32F
    assert lib.named("cublasLtMatmulDescCreate")[0][1:] == [68, 0]  # CUBLAS_COMPUTE_32F, CUDA_R_32F
    assert {a[1]: a[2] for a in lib.named("cublasLtMatmulDescSetAttribute")} == {3: 0, 4: 0, 7: 4, 26: 0, 8: 0x1000}


def test_the_reduction_rule_of_f32_operands_speaks_of_the_output_and_allow_f32_reduction():
    assert L.reduction_problem(4, 1, False, False, F32) == (
        "reduction scheme 1 adds the split-K partial sums in the output (f32), not in a workspace of the compute type (only no split-K, 0, and f32 "
        "partial sums in the workspace, 2, are taken without the opt-in); a declaration may opt in with allow_f32_reduction=True"
    )
    assert "opts in with allow_f32_reduction=True" in L.reduction_problem(4, 2, False, True, F32)
    assert L.reduction_problem(4, 1, True, True, F32) is None and L.reduction_problem(4, 2, False, False, F32) is None


# -- the vetting: TF32 off, no emulation ---------------------------------------------------------------------------------------------------


def test_f32_operands_are_multiplied_as_f32_and_tf32_or_an_emulation_is_refused():
    fma_f32, hmma_tf32, fma_tf32 = 0x80201, 0x40202, 0x40201
    assert L.numerics_problems(fma_f32, 68, F32) == []
    assert L.numerics_problems(0, 68, F32) == []  # the library reports nothing (id 76 on sm_103): held to its compute type alone
    assert L.numerics_problems(hmma_tf32, 68, F32) == [
        "its products read the f32 operands as tf32, not as f32 (CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS 0x40202)"
    ]
    assert len(L.numerics_problems(fma_tf32, 68, F32_NN)) == 1
    assert L.numerics_problems(0x201, 68, F32) == ["its products read the f32 operands as no input type, not as f32 (CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS 0x201)"]
    assert L.numerics_problems(0x20202, 68, F32) == ["its products read the f32 operands as bf16, not as f32 (CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS 0x20202)"]
    for compute, name in ((77, "CUBLAS_COMPUTE_32F_FAST_TF32 (77)"), (78, "CUBLAS_COMPUTE_32F_EMULATED_16BFX9 (78)"), (99, "compute type 99")):
        assert L.numerics_problems(0, compute, F32) == [
            f"cuBLASLt encodes CUBLAS_COMPUTE_32F as {name} in this process, which does not multiply f32 operands as f32 (NVIDIA_TF32_OVERRIDE=1 "
            "turns on TF32, CUBLAS_EMULATE_SINGLE_PRECISION=1 a bf16x9 emulation): an f32 GEMM is not run here"
        ]
        assert len(L.numerics_problems(fma_f32, compute, F32)) == 1
    # the accumulator is asked of every form, and a 16-bit form is asked nothing else (flags 0 are refused there, as in 0.0.5)
    assert L.numerics_problems(0x80101, 68, F32) == ["it does not accumulate in f32"]
    assert L.numerics_problems(0, 68) == ["it does not accumulate in f32"]
    assert L.numerics_problems(0x20202, 77, L.Form("f16", "f32")) == []


def test_the_compute_type_is_read_where_cublaslt_writes_it():
    raw = bytearray(64)
    raw[52] = 0x44
    assert L.compute_type_of(bytes(raw)) == L.CUBLAS_COMPUTE_32F == 68
    raw[52] = 0x4D
    assert L.compute_type_of(bytes(raw)) == 77
    # a bf16 pin of cuBLASLt 12.8.4 (the sm_103 fixture of the Rust crate), and what cublasLtMatmulAlgoInit of 13.1.0 writes for algorithm 76 with
    # f32 types under CUBLAS_COMPUTE_32F and under CUBLAS_COMPUTE_32F_FAST_TF32 (measured on sm_103)
    bf16_pin = "4200000010000000230000000100000000000000000000000100000000000400" "01000000e4d701000e0000000e0000000e000e00440000000000000000000000"
    f32_init = "4c00000000000000000000000100000000000000000000000000000000000000" "0100000034fc0100000000000000000000000000440000000000000000000000"
    tf32_init = f32_init[:104] + "4d" + f32_init[106:]
    assert [L.compute_type_of(bytes.fromhex(h)) for h in (bf16_pin, f32_init, tf32_init)] == [68, 68, 77]


# -- the entry, the digest, the reader -------------------------------------------------------------------------------------------------------


def test_the_entry_of_f32_operands_says_its_types_and_its_opt_in():
    spec = library_specs(declare(nn_bias, layout="NN", allow_f32_reduction=True))[1]
    entry = spec.entry(SMS)
    assert entry["dtypes"] == {"a": "f32", "b": "f32", "bias": "f32", "out": "f32"}
    assert (entry["compute_type"], entry["scale_type"], entry["transa"], entry["lda"]) == ("CUBLAS_COMPUTE_32F", "CUDA_R_32F", "N", 3072)
    assert [p["dtype"] for p in entry["abi"]["params"]] == ["f32"] * 4
    assert entry["allow_f32_reduction"] is True and "allow_f16_reduction" not in entry and "allow_bf16_reduction" not in entry
    assert spec.opt_ins == {"allow_bf16_reduction": False, "allow_f16_reduction": False, "allow_f32_reduction": True}
    validate_entry(built(spec).entry())
    plain = library_specs(declare(tn))[0].entry(SMS)
    assert not any(k.startswith("allow_") for k in plain)


def test_the_opt_in_reaches_the_digest_and_the_types_are_in_the_signature():
    plain = declaration_digest(declare(tn))
    assert _library._canonical(declare(tn, allow_f32_reduction=True))["allow_f32_reduction"] is True
    assert "allow_f32_reduction" not in _library._canonical(declare(tn))
    assert declaration_digest(declare(tn, allow_f32_reduction=True)) != plain
    assert declaration_digest(declare(tn, allow_f32_reduction=False)) == plain

    def tn_f16(a: Tensor[M, 1024, f16], b: Tensor[3072, 1024, f16]) -> Tensor[M, 3072, f32]: ...

    assert declaration_digest(declare(tn_f16)) != plain


# What tilefish 0.0.5 and the f16 forms wrote, computed with the tree before f32 operands existed: the declaration digest, the declaration digest
# of the entry pinned on 30 SMs, and the sha256 of the whole entry (sorted keys). An f32 GEMM changes none of them.
OLD_DIGESTS = {
    "gemm": (
        "4cb08a8eb03dc069aba511ffb00e914b82ef109f8a80812819ac7b05939a449f",
        "4fc7d8c17ee89f694f33330c73c34bf4aa3867caaa9917613e2ba4c719634fb3",
        "ed1c1a22a388aba94b7d2ddb017ebf988daf839dfe73c51fbee34ae63ab777dd",
    ),
    "proj": (
        "088e73c8a099126331c7d8d2f5425ff7950ead539a99680451cad21b05249d66",
        "cacfa1882b7b537b0b2ef56d854216f0770d02c012a9fd529abaae3a78fcddbc",
        "7db875e31353868f0a24043f0962342dfc1a932353298a92ee7d356028e351b1",
    ),
    "up": (
        "980b1c04f0244e8b65490779c642fde0c3c687ac21db35ef0d8321981a43d01b",
        "b3846044cfc76b43129404d9e37283be140453fdabec7156dffff4bda8e2cf28",
        "700e7bd5cdfc8b8f1717e244c44d155867e5e5f1778af4cad3685ed5bbdd97a2",
    ),
    "down": (
        "d9193ee06102bbb539ee61604ab2f25de8a9c6f840500a3941b33b8a4937ae64",
        "362e713b431d0b808d419189abb6375377e84866ff238a7d3ea4def6f5255c73",
        "827748211f2495ff92c8403c3719ebcf0e14eb3e9ea54e193f898507728a3f45",
    ),
}


def test_entries_of_0_0_5_and_of_the_f16_forms_keep_their_bytes_and_digests():
    @tilefish.library("gemm", acc=f32, impl="cublaslt.gemm")
    def gemm(a: Tensor[M, 1024, bf16], b: Tensor[2048, 1024, bf16]) -> Tensor[M, 2048, bf16]: ...

    @tilefish.library("proj", acc=f32, impl="cublaslt.gemm_bias", heuristic="torch", allow_bf16_reduction=True)
    def proj(a: Tensor[M, 512, bf16], b: Tensor[1536, 512, bf16], bias: Tensor[1536, bf16]) -> Tensor[M, 1536, bf16]: ...

    @tilefish.library("up", acc=f32, impl="cublaslt.gemm_bias", layout="NN", allow_f16_reduction=True, workspace_bytes=64 << 20)
    def up(a: Tensor[M, 7168, f16], b: Tensor[7168, 1536, f16], bias: Tensor[1536, f16]) -> Tensor[M, 1536, f16]: ...

    @tilefish.library("down", acc=f32, impl="cublaslt.gemm")
    def down(a: Tensor[M, 3072, f16], b: Tensor[1024, 3072, f16]) -> Tensor[M, 1024, f32]: ...

    for op in (gemm, proj, up, down):
        entry = BuiltLibrary(library_specs(op)[1], "sm_103", ALGO, 120804, 30).entry()
        whole = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        assert (declaration_digest(op), entry["declaration_digest"], whole) == OLD_DIGESTS[op.name], op.name
        assert "allow_f32_reduction" not in entry


def refuses(op, change, message, index=0):
    entry = built(library_specs(op)[index]).entry()
    change(entry)
    with pytest.raises(TilefishManifestError, match=message):
        validate_entry(entry)


def test_the_reader_holds_f32_operands_to_an_f32_result_and_their_own_opt_in():
    op = declare(tn_bias)
    refuses(op, lambda e: e["dtypes"].update(out="bf16", bias="bf16"), "dtypes must be")
    refuses(op, lambda e: e["dtypes"].update(out="f16"), "dtypes must be")
    refuses(op, lambda e: e["dtypes"].update(b="f16"), r"a and b are both bf16, both f16 or both f32")
    refuses(op, lambda e: e["abi"]["params"][1].update(dtype="f16"), r"abi.params\[1\] must be the f32 pointer #1")
    refuses(op, lambda e: e.update(allow_f16_reduction=True), "allow_f16_reduction is the opt-in of f16 operands, and the operands of this entry are f32")
    refuses(op, lambda e: e.update(allow_f32_reduction=False), r"allow_f32_reduction is absent \(the default\) or true, got False")
    refuses(op, lambda e: e.update(allow_f32_reduction=1), r"allow_f32_reduction is absent \(the default\) or true, got 1")

    def bf16_op(a: Tensor[M, 1024, bf16], b: Tensor[3072, 1024, bf16]) -> Tensor[M, 3072, bf16]: ...

    refuses(declare(bf16_op), lambda e: e.update(allow_f32_reduction=True), "allow_f32_reduction is the opt-in of f32 operands, and the operands of this entry are bf16")
    validate_entry(built(library_specs(declare(tn_bias, allow_f32_reduction=True))[0]).entry())


def test_a_manifest_holds_f32_entries_and_a_replay_takes_them(tmp_path, monkeypatch):
    ops = [declare(nn_bias, "proj", layout="NN", allow_f32_reduction=True), declare(tn, "down")]
    library = [built(s) for op in ops for s in library_specs(op)]
    write_manifest([], tmp_path / "m" / "manifest.json", cubins=True, library=library)
    m = Manifest.load(tmp_path / "m")
    assert sorted(e["stem"] for e in m.library_entries) == ["down.M32", "down.M64", "proj.M32", "proj.M64"]
    monkeypatch.setattr(L, "version", lambda: VERSION)
    for op in ops:
        for spec in library_specs(op):
            got = _library.built_from_manifest(m, spec, "sm_103", SMS)
            assert got.algo == ALGO and got.spec is spec
    # the opt-in is part of what the declaration asks for: the entry of a declaration without it is refused for one with it
    with pytest.raises(TilefishManifestError, match="is not what the declaration asks for"):
        _library.built_from_manifest(m, library_specs(declare(tn, "down", allow_f32_reduction=True))[0], "sm_103", SMS)


# -- imported algorithms, the Rust host ------------------------------------------------------------------------------------------------------


def test_an_imported_configuration_is_encoded_with_f32_types(monkeypatch):
    calls = []
    monkeypatch.setattr(_library, "_device_index", lambda i: 0)
    monkeypatch.setattr(_library, "_capability", lambda i: 103)
    monkeypatch.setattr(_library, "sm_count", lambda i: SMS)
    monkeypatch.setattr(L, "version", lambda: VERSION)
    monkeypatch.setattr(L, "encode_algo", lambda cfg, **kw: calls.append(("encode", cfg, kw)) or ALGO)
    monkeypatch.setattr(L, "check_algo", lambda *a, **kw: calls.append(("check", a, kw)) or 0)
    t = AlgoTable("sm_103", SMS, VERSION, {(m, 3072, 1024, True, "NN"): CONFIG for m in (32, 64)})
    spec = library_specs(declare(nn_bias, layout="NN", algos=t, allow_f32_reduction=True))[0]
    got = build_library(spec, arch="sm_103")
    assert got.algo == ALGO and got.entry()["algo_source"] == "imported" and got.entry()["allow_f32_reduction"] is True
    opt_in = {"allow_bf16_reduction": False, "allow_f16_reduction": False, "allow_f32_reduction": True}
    assert calls == [
        ("encode", CONFIG, {"device_index": 0, "form": F32_NN}),
        ("check", (32, 3072, 1024, True, ALGO), {"workspace_bytes": 8_519_680, "device_index": 0, "form": F32_NN, "sm_count_target": None, **opt_in}),
    ]


def test_a_rust_launcher_takes_f32_tensors(tmp_path):
    from tilefish.hosts import rust

    ops = [declare(nn_bias, "proj", layout="NN", allow_f32_reduction=True), declare(tn, "down")]
    write_manifest([], tmp_path / "m" / "manifest.json", cubins=True, library=[built(s) for op in ops for s in library_specs(op)])
    m = Manifest.load(tmp_path / "m")
    source = rust.render(m.check_pin(m.digest), ops=ops)
    assert "a: TensorRef<'a, F32>" in source and "out: TensorMut<'a, F32>" in source and "bias: TensorRef<'a, F32>" in source
    assert "`out[32, 3072] = a[32, 1024] @ b[1024, 3072] + bias[3072]` on f32 operands" in source
    assert "split-K to add its partial sums in the output (`allow_f32_reduction`)" in source
