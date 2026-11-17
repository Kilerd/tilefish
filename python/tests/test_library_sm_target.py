"""A GEMM pinned for a partition of the device's SMs (``@tilefish.library(..., sm_count_target=T)``), with no GPU.

The declaration's ``sm_count_target`` is the SM count cuBLASLt's heuristic chooses for (``CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET``), and every check and
run of the pin sets it too. It reaches the canonical declaration (the digest) and the entry, as an optional field present only when it is set; the
reader holds it to an int from 8 to the entry's ``sm_count``. The entry's ``sm_count`` stays the device's count, which a replay compares with the
device (a green context reports the device's count: measured, ``docs/en/manifest-schema-3.md``, section 9.3). What it does on a GPU, in a green context
too, is ``test_gpu_gemm_sm_target.py``.
"""

import pytest
from test_cublaslt import RecordingLib

import tilefish
from tilefish import Dim, Manifest, Tensor, TilefishBuildError, TilefishDeclarationError, TilefishManifestError, TilefishRuntimeError, f16, f32
from tilefish import _cublaslt as L
from tilefish import _library
from tilefish._library import BuiltLibrary, build_library, built_from_manifest, declaration_digest, library_specs, validate_entry
from tilefish.aot import write_manifest

M = Dim("M", buckets=[32, 64])
VERSION = 130100
SMS = 30
ALGO = "0123456789abcdef" * 8


def gemm(a: Tensor[M, 1024, f16], b: Tensor[1024, 3072, f16], bias: Tensor[3072, f16]) -> Tensor[M, 3072, f16]: ...


def declare(**kw):
    return tilefish.library("g", acc=f32, impl="cublaslt.gemm_bias", layout="NN", **kw)(gemm)


def built(spec, sm_count=SMS):
    return BuiltLibrary(spec, "sm_103", ALGO, VERSION, sm_count)


def test_a_target_is_declared_per_op_and_is_at_least_8():
    assert declare(sm_count_target=16).sm_count_target == 16
    assert declare().sm_count_target is None
    for bad in (7, 0, -16, True, 16.0, "16", 2**31):
        with pytest.raises(TilefishDeclarationError, match="sm_count_target must be None or an int of at least 8"):
            declare(sm_count_target=bad)
    with pytest.raises(TilefishDeclarationError, match="sm_count_target need an impl"):
        tilefish.library("x", sm_count_target=16)


def test_the_target_reaches_the_digest_and_the_entry_only_when_it_is_set():
    plain, sixteen, eight = declare(), declare(sm_count_target=16), declare(sm_count_target=8)
    assert "sm_count_target" not in _library._canonical(plain) and _library._canonical(sixteen)["sm_count_target"] == 16
    assert len({declaration_digest(plain), declaration_digest(sixteen), declaration_digest(eight)}) == 3
    spec = library_specs(sixteen)[0]
    assert spec.sm_count_target == 16
    entry = built(spec).entry()
    assert entry["sm_count_target"] == 16 and entry["sm_count"] == SMS
    validate_entry(entry)
    assert "sm_count_target" not in built(library_specs(plain)[0]).entry()


def refuses(change, message):
    entry = built(library_specs(declare(sm_count_target=16))[0]).entry()
    change(entry)
    with pytest.raises(TilefishManifestError, match=message):
        validate_entry(entry)


def test_the_reader_holds_the_target_to_an_int_from_8_to_the_sm_count():
    for bad in (7, 31, 0, 16.0, "16", True, None):
        refuses(lambda e, bad=bad: e.update(sm_count_target=bad), r"sm_count_target is absent or an int from 8 to sm_count \(30\)")
    entry = built(library_specs(declare(sm_count_target=30))[0]).entry()  # the device's count is a target
    validate_entry(entry)


def test_a_device_with_fewer_sms_than_the_target_is_refused_when_the_algorithm_is_pinned(monkeypatch):
    calls = []
    monkeypatch.setattr(_library, "_device_index", lambda i: 0)
    monkeypatch.setattr(_library, "_capability", lambda i: 103)
    monkeypatch.setattr(L, "version", lambda: VERSION)

    class Pinned:
        algo_hex = ALGO

    monkeypatch.setattr(L, "pin_algo", lambda *a, **kw: calls.append(("pin", kw["sm_count_target"])) or Pinned())
    monkeypatch.setattr(L, "check_algo", lambda *a, **kw: calls.append(("check", kw["sm_count_target"])) or 0)
    spec = library_specs(declare(sm_count_target=16))[0]
    monkeypatch.setattr(_library, "sm_count", lambda i: 30)
    got = build_library(spec, arch="sm_103")
    assert got.entry()["sm_count_target"] == 16 and got.sm_count == 30
    assert calls == [("pin", 16), ("check", 16)]  # the heuristic is asked for 16 SMs, and the pin is checked for them
    monkeypatch.setattr(_library, "sm_count", lambda i: 10)
    with pytest.raises(TilefishBuildError, match="its declaration targets 16 SMs \\(sm_count_target=\\), more than the device has"):
        build_library(spec, arch="sm_103")


def test_a_replay_takes_the_entry_of_its_target_and_refuses_another(tmp_path, monkeypatch):
    sixteen = declare(sm_count_target=16)
    write_manifest([], tmp_path / "m" / "manifest.json", cubins=True, library=[built(s) for s in library_specs(sixteen)])
    m = Manifest.load(tmp_path / "m")
    monkeypatch.setattr(L, "version", lambda: VERSION)
    spec = library_specs(sixteen)[0]
    assert built_from_manifest(m, spec, "sm_103", SMS).spec is spec
    # a declaration with another target (or none) is not what the entry was pinned for
    for other in (declare(), declare(sm_count_target=24)):
        with pytest.raises(TilefishManifestError, match="sm_count_target"):
            built_from_manifest(m, library_specs(other)[0], "sm_103", SMS)


def test_the_binding_sets_the_target_on_the_descriptor_and_refuses_one_below_8(monkeypatch):
    lib = RecordingLib()
    with L._Problem(lib, 8, 16, 32, False, 0, None, None, 16):
        pass
    attrs = {a[1]: (a[2], a[3]) for a in lib.named("cublasLtMatmulDescSetAttribute")}
    assert attrs[15] == (16, 4)  # CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, an int32_t
    lib = RecordingLib()
    with L._Problem(lib, 8, 16, 32, False, 0, None):
        pass
    assert 15 not in {a[1] for a in lib.named("cublasLtMatmulDescSetAttribute")}  # unset unless asked: the library's default
    assert L.describe(1, 2, 3, False, sm_count_target=16)["sm_count_target"] == 16 and L.describe(1, 2, 3, False)["sm_count_target"] is None
    monkeypatch.setattr(L, "_lib", lambda: (_ for _ in ()).throw(AssertionError("the library was reached")))
    for bad in (7, 1, 0, True, 2**31):
        with pytest.raises(TilefishRuntimeError, match="sm_count_target"):
            L.check_algo(8, 16, 32, False, ALGO, workspace_bytes=0, sm_count_target=bad)
    with pytest.raises(TilefishRuntimeError, match="sm_count_target is 7, outside 8"):
        L.pin_algo(8, 16, 32, sm_count_target=7)


def test_the_rust_launcher_says_the_target(tmp_path):
    from tilefish.hosts import rust

    op = declare(sm_count_target=16)
    write_manifest([], tmp_path / "m" / "manifest.json", cubins=True, library=[built(s) for s in library_specs(op)])
    m = Manifest.load(tmp_path / "m")
    source = rust.render(m.check_pin(m.digest), ops=[op])
    assert "The algorithm was chosen for 16 SMs (`sm_count_target`" in source
