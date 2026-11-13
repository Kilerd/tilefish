"""Library operations with an implementation: a cuBLASLt GEMM, declared, specified, written to a manifest and read back, with no GPU.

``@tilefish.library("gemm", acc=f32, impl="cublaslt.gemm")`` over ``(a: Tensor[M, K, bf16], b: Tensor[N, K, bf16]) -> Tensor[M, N, bf16]`` (``impl="cublaslt.gemm_bias"``
adds ``bias: Tensor[N, bf16]``): one binary per bucket of ``M``, ``N`` and ``K``, which is the algorithm cuBLASLt's heuristic ranks first under a fixed
workspace limit, recorded in the manifest's ``library`` section with the cuBLASLt version that chose it. A replay refuses another version. What the
build itself does (the heuristic, the launch) is ``test_cublaslt.py`` and ``test_gpu_gemm.py``.
"""

import copy
import json

import pytest

import tilefish
from tilefish import Dim, Manifest, Tensor, TilefishDeclarationError, TilefishManifestError, TilefishSpecError, bf16, f32
from tilefish import _cublaslt, _library
from tilefish._build import write_manifest_doc
from tilefish._library import BuiltLibrary, build_library, built_from_manifest, declaration_digest, library_specs, validate_entry
from tilefish._manifest import manifest_digest
from tilefish.aot import write_manifest

M = Dim("M", buckets=[32, 64])
Kb = Dim("Kb", buckets=[1024, 2048])
ALGO = "0123456789abcdef" * 8  # 64 bytes of hex: not an algorithm anyone chose, but a well-formed one
VERSION = 120804
SMS = 30  # the multiprocessors of the device a pin is made on (a B300 MIG slice of 1g has 30)


@tilefish.library("gemm", acc=f32, impl="cublaslt.gemm")
def gemm(a: Tensor[M, 1024, bf16], b: Tensor[2048, 1024, bf16]) -> Tensor[M, 2048, bf16]: ...


@tilefish.library("gemm_bias", acc=f32, impl="cublaslt.gemm_bias", workspace_bytes=1 << 20)
def gemm_bias(a: Tensor[M, Kb, bf16], b: Tensor[2048, Kb, bf16], bias: Tensor[2048, bf16]) -> Tensor[M, 2048, bf16]: ...


def built(spec, arch="sm_103", algo=ALGO, version=VERSION, sm_count=SMS):
    return BuiltLibrary(spec, arch, algo, version, sm_count)


# -- the declaration ------------------------------------------------------------------------------------------------------------------------


def test_a_gemm_is_a_library_op_with_an_implementation_and_a_workspace_limit():
    assert gemm.kind == "library" and gemm.impl == "cublaslt.gemm" and gemm.workspace_bytes == 8_519_680 and gemm.acc is f32
    assert gemm_bias.impl == "cublaslt.gemm_bias" and gemm_bias.workspace_bytes == 1 << 20


def test_a_library_op_without_an_implementation_is_a_signature_only():
    @tilefish.library("opaque", acc=f32)
    def opaque(a: Tensor[M, 8, bf16]) -> Tensor[M, 8, bf16]: ...

    assert opaque.impl is None and opaque.workspace_bytes is None
    with pytest.raises(TilefishDeclarationError, match="workspace_bytes needs an impl"):
        tilefish.library("x", workspace_bytes=1024)
    with pytest.raises(TilefishSpecError, match="has no implementation"):
        library_specs(opaque)


def declare(fn, impl="cublaslt.gemm", **kw):
    return tilefish.library("g", acc=f32, impl=impl, **kw)(fn)


def test_an_unknown_impl_is_refused():
    with pytest.raises(TilefishDeclarationError, match="impl must be one of"):
        declare(gemm.fn, impl="nope")


def _one_arg(a: Tensor[M, 1024, bf16]) -> Tensor[M, 1024, bf16]: ...


def _two_args(a: Tensor[M, 1024, bf16], b: Tensor[2048, 1024, bf16]) -> Tensor[M, 2048, bf16]: ...


def _no_return(a: Tensor[M, 1024, bf16], b: Tensor[2048, 1024, bf16]): ...


def _f32_operand(a: Tensor[M, 1024, f32], b: Tensor[2048, 1024, bf16]) -> Tensor[M, 2048, bf16]: ...


def _k_mismatch(a: Tensor[M, 1024, bf16], b: Tensor[2048, 2048, bf16]) -> Tensor[M, 2048, bf16]: ...


def _wrong_result(a: Tensor[M, 1024, bf16], b: Tensor[2048, 1024, bf16]) -> Tensor[M, 1024, bf16]: ...


def _bias_rank2(a: Tensor[M, 1024, bf16], b: Tensor[2048, 1024, bf16], bias: Tensor[1, 2048, bf16]) -> Tensor[M, 2048, bf16]: ...


def _bias_wrong_size(a: Tensor[M, 1024, bf16], b: Tensor[2048, 1024, bf16], bias: Tensor[1024, bf16]) -> Tensor[M, 2048, bf16]: ...


def test_the_signature_must_be_a_gemm_of_bf16_operands():
    with pytest.raises(TilefishDeclarationError, match="takes 2 input tensors"):
        declare(_one_arg)
    with pytest.raises(TilefishDeclarationError, match="takes 3 input tensors"):
        declare(_two_args, "cublaslt.gemm_bias")
    with pytest.raises(TilefishDeclarationError, match="declare a return type"):
        declare(_no_return)
    with pytest.raises(TilefishDeclarationError, match="a must be a bf16 tensor of rank 2"):
        declare(_f32_operand)
    with pytest.raises(TilefishDeclarationError, match=r"a is \[M, K\] and b is \[N, K\]: their last axes must be the same"):
        declare(_k_mismatch)
    with pytest.raises(TilefishDeclarationError, match=r"the result must be \[M, N\] = \[M, 2048\]"):
        declare(_wrong_result)
    with pytest.raises(TilefishDeclarationError, match="bias must be a bf16 tensor of rank 1"):
        declare(_bias_rank2, "cublaslt.gemm_bias")
    with pytest.raises(TilefishDeclarationError, match=r"and bias \[N\] = \[2048\]"):
        declare(_bias_wrong_size, "cublaslt.gemm_bias")


def _k_1020(a: Tensor[M, 1020, bf16], b: Tensor[2048, 1020, bf16]) -> Tensor[M, 2048, bf16]: ...


def _n_2044(a: Tensor[M, 1024, bf16], b: Tensor[2044, 1024, bf16]) -> Tensor[M, 2044, bf16]: ...


def _k_9(a: Tensor[M, 9, bf16], b: Tensor[96, 9, bf16], bias: Tensor[96, bf16]) -> Tensor[M, 96, bf16]: ...


def test_every_dimension_is_static_or_bucketed_and_n_and_k_may_be_any_size():
    free = Dim("free")

    def f(a: Tensor[free, 1024, bf16], b: Tensor[2048, 1024, bf16]) -> Tensor[free, 2048, bf16]: ...

    with pytest.raises(TilefishDeclarationError, match="depends on the dynamic dimension free without buckets"):
        declare(f)
    # rows that do not start on 16 bytes: the library says which algorithms can take those leading dimensions (cublasLtMatmulAlgoCheck)
    k = library_specs(declare(_k_1020))[0]
    assert (k.k, k.entry(SMS)["lda"], k.entry(SMS)["ldb"]) == (1020, 1020, 1020)
    n = library_specs(declare(_n_2044))[0]
    assert (n.n, n.entry(SMS)["ldc"], n.entry(SMS)["ldd"]) == (2044, 2044, 2044)
    nine = library_specs(declare(_k_9, "cublaslt.gemm_bias"))[1]
    assert (nine.m, nine.n, nine.k) == (64, 96, 9)
    validate_entry(built(nine).entry())
    with pytest.raises(TilefishDeclarationError, match="workspace_bytes must be an int from 1"):
        declare(_two_args, workspace_bytes=0)
    with pytest.raises(TilefishDeclarationError, match="accumulates in f32"):
        tilefish.library("g", acc=tilefish.bf16, impl="cublaslt.gemm")(_two_args)


def test_a_dimension_that_is_divisible_by_8_is_a_valid_n_or_k():
    N = Dim("N", divisible_by=8, buckets=[1024, 3072])

    @tilefish.library("g", acc=f32, impl="cublaslt.gemm")
    def g(a: Tensor[M, 1024, bf16], b: Tensor[N, 1024, bf16]) -> Tensor[M, N, bf16]: ...

    assert [s.stem for s in library_specs(g)] == ["g.M32.N1024", "g.M32.N3072", "g.M64.N1024", "g.M64.N3072"]


def test_a_library_op_cannot_take_an_out_parameter():
    with pytest.raises(TilefishDeclarationError, match="takes 2 input tensors"):

        @tilefish.library("g", acc=f32, impl="cublaslt.gemm")
        def g(a: Tensor[M, 1024, bf16], b: tilefish.Out[Tensor[2048, 1024, bf16]]) -> Tensor[M, 2048, bf16]: ...


def test_the_declaration_digest_covers_the_name_the_implementation_the_workspace_and_the_signature():
    base = declaration_digest(gemm)
    assert base == declaration_digest(gemm) and len(base) == 64
    other_ws = tilefish.library("gemm", acc=f32, impl="cublaslt.gemm", workspace_bytes=4096)(gemm.fn)
    assert declaration_digest(other_ws) != base
    renamed = tilefish.library("gemm2", acc=f32, impl="cublaslt.gemm")(gemm.fn)
    assert declaration_digest(renamed) != base
    assert declaration_digest(gemm_bias) != base


# -- the specs ------------------------------------------------------------------------------------------------------------------------------


def test_one_spec_per_bucket_combination_with_the_stem_of_a_kernel():
    specs = library_specs(gemm_bias)
    assert [s.stem for s in specs] == ["gemm_bias.Kb1024.M32", "gemm_bias.Kb1024.M64", "gemm_bias.Kb2048.M32", "gemm_bias.Kb2048.M64"]
    s = specs[1]
    assert (s.m, s.n, s.k, s.bias, s.impl, s.workspace_bytes) == (64, 2048, 1024, True, "cublaslt.gemm_bias", 1 << 20)
    assert s.buckets == {"Kb": 1024, "M": 64} and s.dims == {"Kb": {"buckets": [1024, 2048], "bucket": 1024}, "M": {"buckets": [32, 64], "bucket": 64}}
    assert len(library_specs(gemm)) == 2


def test_the_entry_of_a_spec_says_the_problem_as_cublaslt_is_called():
    s = library_specs(gemm)[0]
    entry = s.entry(SMS)
    assert set(entry) == set(_library.ENTRY_KEYS) - {"arch", "algo", "cublaslt_version"}
    assert (entry["stem"], entry["op"], entry["impl"], entry["epilogue"]) == ("gemm.M32", "gemm", "cublaslt.gemm", "none")
    assert (entry["m"], entry["n"], entry["k"]) == (32, 2048, 1024) and (entry["lda"], entry["ldb"], entry["ldc"], entry["ldd"]) == (1024, 1024, 2048, 2048)
    assert (entry["transa"], entry["transb"], entry["compute_type"], entry["scale_type"]) == ("T", "N", "CUBLAS_COMPUTE_32F", "CUDA_R_32F")
    assert entry["dtypes"] == {"a": "bf16", "b": "bf16", "out": "bf16"} and entry["workspace_bytes"] == 8_519_680
    assert [(p["name"], p["access"], p["shape"]) for p in entry["abi"]["params"]] == [("a", "in", ["M", 1024]), ("b", "in", [2048, 1024]), ("out", "out", ["M", 2048])]
    assert entry["buckets"] == {"M": 32} and entry["declaration_digest"] == declaration_digest(gemm, SMS) != declaration_digest(gemm) and entry["sm_count"] == SMS
    pres = entry["preconditions"]
    assert [p for p in pres if p["kind"] == "pointer_alignment"] == [{"kind": "pointer_alignment", "arg": n, "index": i, "bytes": 16} for i, n in enumerate(["a", "b", "out"])]
    assert {"kind": "static_size", "arg": "b", "index": 1, "axis": 0, "value": 2048} in pres and {"kind": "bucket", "dim": "M", "value": 32, "buckets": [32, 64]} in pres
    bias = library_specs(gemm_bias)[0].entry(SMS)
    assert bias["epilogue"] == "bias" and bias["dtypes"]["bias"] == "bf16" and [p["name"] for p in bias["abi"]["params"]] == ["a", "b", "bias", "out"]


def test_the_entry_says_what_the_binding_calls_cublaslt_with():
    for spec in library_specs(gemm) + library_specs(gemm_bias):
        entry, d = spec.entry(SMS), _cublaslt.describe(spec.m, spec.n, spec.k, spec.bias)
        for key in ("transa", "transb", "compute_type", "scale_type", "lda", "ldb", "ldc", "ldd", "epilogue"):
            assert entry[key] == d[key], key
        assert (d["cublas_m"], d["cublas_n"], d["cublas_k"]) == (spec.n, spec.m, spec.k)  # tilefish's b is cuBLAS's A: m and n swap


# -- the manifest ---------------------------------------------------------------------------------------------------------------------------


def write(tmp_path, *lib, kernels=(), **kw):
    path = tmp_path / "m" / "manifest.json"
    write_manifest(kernels, path, cubins=True, library=lib, **kw)
    return Manifest.load(tmp_path / "m")


def test_a_manifest_has_a_library_section_without_files(tmp_path):
    specs = library_specs(gemm)
    m = write(tmp_path, *(built(s) for s in specs))
    doc = json.loads((tmp_path / "m" / "manifest.json").read_text())
    assert doc["schema"] == 3 and doc["kernels"] == [] and [e["stem"] for e in doc["library"]] == ["gemm.M32", "gemm.M64"]
    assert [p.name for p in (tmp_path / "m").iterdir()] == ["manifest.json"]
    assert m.library_entry("gemm.M32", "sm_103")["algo"] == ALGO and m.library_entry("gemm.M32", "sm_90") is None
    assert [e["stem"] for e in m.library_entries] == ["gemm.M32", "gemm.M64"] and m.artifacts == []
    assert m.digest == manifest_digest(doc)


def test_the_digest_covers_the_library_section(tmp_path):
    m = write(tmp_path, built(library_specs(gemm)[0]))
    doc = json.loads((tmp_path / "m" / "manifest.json").read_text())
    for change in (
        lambda e: e.update(algo="f" * 128),
        lambda e: e.update(cublaslt_version=VERSION + 1),
        lambda e: e.update(workspace_bytes=1),
        lambda e: e.update(arch="sm_90"),
        lambda e: e["preconditions"].pop(),
    ):
        other = copy.deepcopy(doc)
        change(other["library"][0])
        assert manifest_digest(other) != m.digest
    other = copy.deepcopy(doc)
    other["library"].append(copy.deepcopy(other["library"][0]))
    other["library"][1]["stem"] = "gemm.M64"
    assert manifest_digest(other) != m.digest
    reordered = copy.deepcopy(other)
    reordered["library"].reverse()
    assert manifest_digest(reordered) == manifest_digest(other)  # the entries are sorted by (stem, arch)
    assert manifest_digest({"schema": 3, "kernels": [], "library": []}) != manifest_digest({"schema": 2, "kernels": []})


def test_an_export_merges_library_entries_like_kernels(tmp_path):
    a, b = library_specs(gemm)
    write(tmp_path, built(a))
    write(tmp_path, built(b), merge=True)
    doc = json.loads((tmp_path / "m" / "manifest.json").read_text())
    assert [e["stem"] for e in doc["library"]] == ["gemm.M32", "gemm.M64"]
    with pytest.raises(TilefishManifestError, match="already has a different library entry for 'gemm.M32' on sm_103 \\(algo differs\\)"):
        write(tmp_path, built(a, algo="f" * 128), merge=True)
    write(tmp_path, built(a, algo="f" * 128), merge=True, overwrite=True)
    assert json.loads((tmp_path / "m" / "manifest.json").read_text())["library"][0]["algo"] == "f" * 128
    with pytest.raises(Exception, match="one build per"):
        write_manifest_doc([], tmp_path / "x.json", library=[built(a), built(a)])


def test_a_replay_takes_an_entry_that_is_what_the_declaration_asks_for(tmp_path, monkeypatch):
    m = write(tmp_path, *(built(s) for s in library_specs(gemm)))
    monkeypatch.setattr(_cublaslt, "version", lambda: VERSION)
    spec = library_specs(gemm)[1]
    got = built_from_manifest(m, spec, "sm_103", SMS)
    assert got.algo == ALGO and got.cublaslt_version == VERSION and got.spec is spec and got.stem == "gemm.M64" and got.workspace_bytes == 8_519_680
    assert got.entry() == m.library_entry("gemm.M64", "sm_103")


def test_a_replay_refuses_a_device_with_another_number_of_sms(tmp_path, monkeypatch):
    """A pin is chosen for the SM count of its device (a MIG slice has fewer than the whole GPU, a stream-K algorithm partitions by it): another
    count is refused, before anything else of the entry, with what to do about it."""
    m = write(tmp_path, built(library_specs(gemm)[0]))
    monkeypatch.setattr(_cublaslt, "version", lambda: VERSION + 100)  # the SM count is the first thing a replay compares
    for other in (SMS - 1, SMS + 1, 148):
        with pytest.raises(
            TilefishManifestError,
            match=rf"pinned the cuBLASLt algorithm of 'gemm.M32' \(sm_103\) on a device with {SMS} SMs, but this device has {other}: .*re-export the manifest on this device",
        ):
            built_from_manifest(m, library_specs(gemm)[0], "sm_103", other)


def test_the_sm_count_is_in_the_entry_its_declaration_digest_and_the_manifest_digest(tmp_path):
    spec = library_specs(gemm)[0]
    a, b = built(spec).entry(), built(spec, sm_count=148).entry()
    assert a["sm_count"] == SMS and b["sm_count"] == 148 and list(a).index("sm_count") == list(a).index("workspace_bytes") + 1
    assert a["declaration_digest"] == spec.pinned_digest(SMS) == declaration_digest(gemm, SMS)
    assert len({a["declaration_digest"], b["declaration_digest"], spec.declaration_digest}) == 3  # the pin of each device is its own declaration
    assert _library.ENTRY_KEYS.index("sm_count") == _library.ENTRY_KEYS.index("cublaslt_version") + 1
    m30 = write(tmp_path / "a", built(spec))
    m148 = write(tmp_path / "b", built(spec, sm_count=148))
    assert m30.digest != m148.digest
    with pytest.raises(TilefishManifestError, match=r"already has a different library entry for 'gemm.M32' on sm_103 \((declaration_digest|sm_count) differs\)"):
        write(tmp_path / "a", built(spec, sm_count=148), merge=True)  # a re-export on another device does not silently replace the pin


def test_a_replay_refuses_a_cublaslt_of_another_version(tmp_path, monkeypatch):
    m = write(tmp_path, built(library_specs(gemm)[0]))
    monkeypatch.setattr(_cublaslt, "version", lambda: VERSION + 100)
    with pytest.raises(
        TilefishManifestError,
        match=rf"the manifest pinned the cuBLASLt algorithm of 'gemm.M32' \(sm_103\) with cuBLASLt {VERSION}, but this process has cuBLASLt {VERSION + 100}: "
        r"an algorithm id names other kernels in another version, so the pin is refused",
    ):
        built_from_manifest(m, library_specs(gemm)[0], "sm_103", SMS)


def test_a_replay_refuses_a_missing_entry_and_an_entry_that_differs_from_the_declaration(tmp_path, monkeypatch):
    monkeypatch.setattr(_cublaslt, "version", lambda: VERSION)
    first, second = library_specs(gemm)
    m = write(tmp_path, built(first))
    with pytest.raises(TilefishManifestError, match=r"has no library entry for 'gemm.M64' on sm_103 \(it has \['gemm.M32'\]\)"):
        built_from_manifest(m, second, "sm_103", SMS)
    with pytest.raises(TilefishManifestError, match=r"has no library entry for 'gemm.M32' on sm_90 \(it has sm_103\)"):
        built_from_manifest(m, first, "sm_90", SMS)
    # the same stem, another declaration: a different workspace limit, another weight shape
    other_ws = tilefish.library("gemm", acc=f32, impl="cublaslt.gemm", workspace_bytes=4096)(gemm.fn)
    with pytest.raises(TilefishManifestError, match="first difference at workspace_bytes: the manifest has 8519680, the declaration gives 4096"):
        built_from_manifest(m, library_specs(other_ws)[0], "sm_103", SMS)

    @tilefish.library("gemm", acc=f32, impl="cublaslt.gemm")
    def wider(a: Tensor[M, 1024, bf16], b: Tensor[4096, 1024, bf16]) -> Tensor[M, 4096, bf16]: ...

    with pytest.raises(TilefishManifestError, match="first difference at abi.params\\[1\\].shape\\[0\\]|first difference at n"):
        built_from_manifest(m, library_specs(wider)[0], "sm_103", SMS)


def test_the_rust_emitter_writes_a_launcher_for_a_library_entry(tmp_path):
    """One struct per stem, wrapping the crate's ``Gemm`` (``test_rust_host_schema3.py`` has the rest)."""
    from tilefish.hosts import rust

    m = write(tmp_path, built(library_specs(gemm)[0]))
    source = rust.render(m.check_pin(m.digest), ops=[gemm])
    assert "pub struct GemmM32 {\n    gemm: Gemm,\n}" in source and "manifest.gemm(ctx, Self::STEM, Self::BUCKETS" in source


def test_pinning_asks_the_device_of_the_architecture(monkeypatch):
    """A pin is per device: the heuristic runs on the device, which must be of the architecture the entry is for, and the entry records its SM count."""
    spec = library_specs(gemm)[0]
    calls = []

    class FakePinned:
        algo_hex = ALGO

    monkeypatch.setattr(_library, "_device_index", lambda i: 0)
    monkeypatch.setattr(_library, "_capability", lambda i: 90)
    monkeypatch.setattr(_cublaslt, "pin_algo", lambda *a, **kw: (calls.append((a, kw)) or FakePinned()))
    monkeypatch.setattr(_cublaslt, "check_algo", lambda *a, **kw: calls.append(("check", a, kw)) or 0)
    monkeypatch.setattr(_cublaslt, "version", lambda: VERSION)
    monkeypatch.setattr(_library, "sm_count", lambda i: 132)
    got = build_library(spec, arch="sm_90")
    assert got.algo == ALGO and got.arch == "sm_90" and got.cublaslt_version == VERSION and got.sm_count == 132 and got.entry()["sm_count"] == 132
    assert calls == [
        ((32, 2048, 1024, False), {"workspace_bytes": 8_519_680, "device_index": 0, "heuristic": "tilefish", "allow_bf16_reduction": False}),
        # the pin passes the checks of a run before it is recorded
        ("check", (32, 2048, 1024, False, ALGO), {"workspace_bytes": 8_519_680, "device_index": 0, "allow_bf16_reduction": False}),
    ]
    with pytest.raises(tilefish.TilefishBuildError, match="cannot pin 'gemm.M32' for sm_103: cuBLASLt chooses an algorithm for the device it runs on, and device 0 is sm_90"):
        build_library(spec, arch="sm_103")


# -- reading an entry back, strictly ---------------------------------------------------------------------------------------------------------


def entry_of(op=gemm, index=0):
    return built(library_specs(op)[index]).entry()


def refuses(change, message, op=gemm):
    entry = entry_of(op)
    change(entry)
    with pytest.raises(TilefishManifestError, match=message):
        validate_entry(entry)


def test_a_well_formed_entry_validates():
    validate_entry(entry_of(gemm))
    for i in range(4):
        validate_entry(entry_of(gemm_bias, i))


def test_the_fields_are_exactly_these():
    refuses(lambda e: e.pop("algo"), "must have exactly the fields.*missing \\['algo'\\]")
    refuses(lambda e: e.update(extra=1), "unknown \\['extra'\\]")
    with pytest.raises(TilefishManifestError, match="is not an object"):
        validate_entry(7)
    refuses(lambda e: e.update(stem="1x"), "stem must look like")
    refuses(lambda e: e.update(op="a b"), "op must be an identifier")
    refuses(lambda e: e.update(arch="sm_9"), "arch must look like 'sm_103'")
    refuses(lambda e: e.update(arch="sm_103a"), "arch must look like 'sm_103'")
    refuses(lambda e: e.update(impl="cublaslt.nope"), "impl must be one of")


@pytest.mark.parametrize("key", ["m", "n", "k", "lda", "ldb", "ldc", "ldd"])
@pytest.mark.parametrize("bad", [0, -1, 2**31, 1.0, "3", True, None])
def test_the_sizes_are_ints_from_1_to_2_31_minus_1(key, bad):
    refuses(lambda e: e.update({key: bad}), f"{key} must be an int from 1 to 2147483647")


def test_the_problem_is_a_tn_gemm_and_its_redundant_fields_must_agree():
    refuses(lambda e: e.update(lda=3), "the leading dimensions must be lda = ldb = k and ldc = ldd = n")
    refuses(lambda e: e.update(ldd=1), "the leading dimensions must be")
    refuses(lambda e: e.update(transa="N"), "transa and transb must be 'T' and 'N'")
    refuses(lambda e: e.update(transb="T"), "transa and transb must be 'T' and 'N'")
    refuses(lambda e: e.update(compute_type="CUBLAS_COMPUTE_16F"), "compute_type and scale_type must be")
    refuses(lambda e: e.update(scale_type="CUDA_R_16F"), "compute_type and scale_type must be")
    refuses(lambda e: e.update(epilogue="bias"), "the epilogue of cublaslt.gemm must be 'none'")
    refuses(lambda e: e.update(epilogue="none"), "the epilogue of cublaslt.gemm_bias must be 'bias'", op=gemm_bias)
    refuses(lambda e: e["dtypes"].update(a="f32"), "dtypes must be")
    refuses(lambda e: e["dtypes"].update(bias="bf16"), "dtypes must be")
    refuses(lambda e: e["dtypes"].pop("out"), "dtypes must be")
    refuses(lambda e: e.update(m=33, ), "abi.params\\[0\\].shape\\[0\\] is 'M' = 32, but m, n and k say 33")
    refuses(lambda e: e.update(k=1536, lda=1536, ldb=1536), "abi.params\\[0\\].shape\\[1\\] is 1024 = 1024, but m, n and k say 1536")


def test_the_pin_and_the_provenance_are_exactly_typed():
    refuses(lambda e: e.update(algo=ALGO[:-1]), "algo must be 128 lowercase hex digits")
    refuses(lambda e: e.update(algo=ALGO.upper()), "algo must be 128 lowercase hex digits")
    refuses(lambda e: e.update(algo="g" * 128), "algo must be 128 lowercase hex digits")
    refuses(lambda e: e.update(algo=ALGO + "\n"), "algo must be 128 lowercase hex digits")
    refuses(lambda e: e.update(cublaslt_version=0), "cublaslt_version must be a positive int")
    refuses(lambda e: e.update(cublaslt_version="120804"), "cublaslt_version must be a positive int")
    refuses(lambda e: e.update(cublaslt_version=True), "cublaslt_version must be a positive int")
    refuses(lambda e: e.update(workspace_bytes=0), "workspace_bytes must be an int from 1 to 2\\*\\*40")
    refuses(lambda e: e.update(workspace_bytes=2**40 + 1), "workspace_bytes must be an int from 1 to 2\\*\\*40")
    refuses(lambda e: e.update(workspace_bytes=1.5), "workspace_bytes must be an int")
    refuses(lambda e: e.update(declaration_digest="0" * 63), "declaration_digest must be 64 lowercase hex digits")


@pytest.mark.parametrize("bad", [0, -1, 30.0, True, "30", None])
def test_the_sm_count_is_a_positive_int(bad):
    refuses(lambda e: e.update(sm_count=bad), "sm_count must be a positive int")
    refuses(lambda e: e.pop("sm_count"), r"missing \['sm_count'\]")
    entry = entry_of()
    entry["sm_count"] = 2**40  # any int >= 1: the reader has no other bound
    validate_entry(entry)


@pytest.mark.parametrize("bad", [0.0, False])
def test_the_index_of_an_operand_is_an_int_not_a_float_or_a_bool(bad):
    """``0.0 == 0`` and ``True == 1`` in Python: the comparison with the position alone let them through (Rust refuses them)."""
    refuses(lambda e: e["abi"]["params"][0].update(index=bad), r"must be the bf16 pointer #0")
    refuses(lambda e: e["abi"]["params"][1].update(index=True if bad is False else 1.0), r"must be the bf16 pointer #1")
    refuses(lambda e: e["dims"]["M"].update(bucket=32.0), r"dims.M.bucket must be one of its buckets")


def test_the_operands_are_a_b_bias_out_in_this_order():
    refuses(lambda e: e["abi"]["params"].pop(), "abi.params must list 3 pointers")
    refuses(lambda e: e["abi"].update(extra=1), "unknown fields")
    refuses(lambda e: e["abi"]["params"][0].update(kind="scalar"), "must be the bf16 pointer #0 with access in")
    refuses(lambda e: e["abi"]["params"][0].update(dtype="f32"), "must be the bf16 pointer #0")
    refuses(lambda e: e["abi"]["params"][2].update(access="in"), "must be the bf16 pointer #2 with access out")
    refuses(lambda e: e["abi"]["params"][1].update(index=5), "must be the bf16 pointer #1")
    refuses(lambda e: e["abi"]["params"][1].update(name="a"), "the name 'a' is used twice")
    refuses(lambda e: e["abi"]["params"][0].update(shape=["M"]), "shape must have 2 entries")
    refuses(lambda e: e["abi"]["params"][0].update(shape=["M", "K"]), "mentions a dimension that has no bucket")
    refuses(lambda e: e["abi"]["params"][0].update(shape=["M", 1.5]), "must be an int or a dimension expression")
    refuses(lambda e: e["abi"]["params"][0].update(shape=["M; drop()", 1024]), "dimension grammar")
    refuses(lambda e: e["abi"]["params"][0].pop("access"), "must have exactly the fields|is missing")


def test_the_buckets_and_the_table_of_dimensions_are_checked():
    refuses(lambda e: e["buckets"].update(M=48), "bucket precondition for M disagrees with buckets|abi.params\\[0\\].shape\\[0\\] is 'M' = 48")
    refuses(lambda e: e["buckets"].update(Z=3), "every bucket needs exactly one bucket precondition")
    refuses(lambda e: e["dims"]["M"].update(bucket=64), "dims.M.bucket must be one of its buckets and the bucket of this entry")
    refuses(lambda e: e["dims"].pop("M"), "every bucketed dimension needs its bucket in dims: M has none")
    refuses(lambda e: e["dims"]["M"].update(buckets=[64, 32]), "dims.M.buckets must be a sorted list")
    refuses(lambda e: e["dims"]["M"].update(divisible_by=0), "divisible_by must be an int >= 1")
    refuses(lambda e: e.update(stem="gemm"), "stem 'gemm' does not match op 'gemm' and buckets {'M': 32}")


def test_the_preconditions_are_the_promises_of_a_kernel():
    refuses(lambda e: e["preconditions"].pop(0), "every pointer needs exactly one pointer_alignment precondition")
    refuses(lambda e: e["preconditions"][0].update(bytes=8), "pointer_alignment must be 16 bytes, once per pointer")
    refuses(lambda e: e["preconditions"][0].update(arg="b"), "pointer_alignment must be 16 bytes, once per pointer")
    refuses(lambda e: e["preconditions"].append(copy.deepcopy(e["preconditions"][0])), "pointer_alignment must be 16 bytes, once per pointer")
    refuses(
        lambda e: e["preconditions"].append({"kind": "max_elements", "arg": "a", "index": 0, "value": 2**31 - 1, "shape": ["M", "1024"]}),
        "a max_elements precondition does not belong in a library entry",
    )
    refuses(lambda e: e["preconditions"].append({"kind": "static_size", "arg": "b", "index": 1, "axis": 0, "value": 7}), "static_size does not match")
    refuses(lambda e: e["preconditions"].append({"kind": "static_size", "arg": "zzz", "index": 9, "axis": 0, "value": 7}), "static_size does not match")
    refuses(lambda e: e["preconditions"].append({"kind": "nonsense"}), "precondition kind must be one of")
    refuses(lambda e: e.update(preconditions="x"), "preconditions must be a list")


def test_a_manifest_with_a_bad_library_entry_does_not_load(tmp_path):
    write(tmp_path, built(library_specs(gemm)[0]))
    doc = json.loads((tmp_path / "m" / "manifest.json").read_text())
    for change, message in (
        (lambda d: d["library"][0].update(algo="zz"), "algo must be 128 lowercase hex digits"),
        (lambda d: d["library"].append(copy.deepcopy(d["library"][0])), "twice"),
        (lambda d: d.update(library={"a": 1}), "not a tilefish manifest"),
        (lambda d: d["library"].append(5), "is not an object"),
    ):
        edited = copy.deepcopy(doc)
        change(edited)
        (tmp_path / "m" / "manifest.json").write_text(json.dumps(edited))
        with pytest.raises(TilefishManifestError, match=message):
            Manifest.load(tmp_path / "m")


def test_a_library_entry_and_a_kernel_may_not_share_a_stem_and_architecture(tmp_path):
    from binary_fakes import fake_binary_build, make_ptx_op, write_ptx
    from tilefish.aot import compile_specs

    op = make_ptx_op(write_ptx(tmp_path), name="axpy")
    (spec,) = compile_specs(op)
    m = write(tmp_path, built(library_specs(gemm)[0]), kernels=[fake_binary_build(spec, op.binary, arch="sm_103")])
    assert len(m.artifacts) == 1 and len(m.library_entries) == 1
    doc = json.loads((tmp_path / "m" / "manifest.json").read_text())
    doc["library"][0]["stem"] = "axpy"
    doc["library"][0]["op"] = "axpy"
    doc["library"][0]["buckets"] = {}
    # (the library entry is no longer a valid entry for that stem, but the loader says what is wrong first: the pair is taken twice)
    (tmp_path / "m" / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError):
        Manifest.load(tmp_path / "m")
