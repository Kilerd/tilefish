"""How a pinned cuBLASLt GEMM's algorithm is chosen, with no GPU: the heuristic posture (``heuristic=``), the opt-in to bf16 split-K partial
sums (``allow_bf16_reduction=``) and imported algorithms (``algos=``, :class:`tilefish.AlgoTable`).

All three are part of the declaration (its digest), reach the manifest's ``library`` entry as fields that are there only when they are not the
default (so a declaration without them writes what tilefish 0.0.4 wrote), and are checked by the strict reader. What they do on a GPU (that the
``"torch"`` posture pins what torch runs, that an imported algorithm runs, that bf16 partial sums need the opt-in) is ``test_gpu_cublaslt.py`` and
``test_gpu_gemm.py``.
"""

import json

import pytest

import tilefish
from tilefish import AlgoTable, Dim, Manifest, Tensor, TilefishBuildError, TilefishDeclarationError, TilefishManifestError, bf16, f32
from tilefish import _cublaslt, _library
from tilefish._library import BuiltLibrary, build_library, built_from_manifest, declaration_digest, library_specs, validate_entry
from tilefish.aot import write_manifest

M = Dim("M", buckets=[32, 64])
VERSION = 120804
SMS = 30
ALGO = "0123456789abcdef" * 8
OTHER = "fedcba9876543210" * 8
CONFIG = {
    "id": 66, "tile_id": 80, "splitk_num": 1, "reduction_scheme": 0, "cta_swizzling": 0, "custom_option": 0, "stages_id": 35,
    "inner_shape_id": 0, "cluster_shape_id": 3,
}  # fmt: skip
SPLIT_BF16 = {**CONFIG, "splitk_num": 2, "reduction_scheme": 4}


def plain(a: Tensor[M, 1024, bf16], b: Tensor[2048, 1024, bf16]) -> Tensor[M, 2048, bf16]: ...


def biased(a: Tensor[M, 1024, bf16], b: Tensor[2048, 1024, bf16], bias: Tensor[2048, bf16]) -> Tensor[M, 2048, bf16]: ...


def declare(fn=biased, name="g", **kw):
    impl = "cublaslt.gemm_bias" if fn is biased else "cublaslt.gemm"
    return tilefish.library(name, acc=f32, impl=impl, **kw)(fn)


def table(rows=None, arch="sm_103", sm_count=SMS, version=VERSION):
    rows = rows if rows is not None else {(m, 2048, 1024, True): CONFIG for m in (32, 64)}
    return AlgoTable(arch, sm_count, version, rows)


# -- the declaration ------------------------------------------------------------------------------------------------------------------------


def test_the_posture_is_tilefish_unless_it_says_torch_and_torch_has_its_own_workspace_limits():
    op = declare()
    assert (op.heuristic, op.allow_bf16_reduction, op.algos, op.workspace_bytes) == ("tilefish", False, None, 8_519_680)
    # torch's addmm asks the heuristic under its cuBLASLt workspace (1 MiB); torch's mm runs cublasGemmEx with its cuBLAS workspace
    assert declare(heuristic="torch").workspace_bytes == 1_048_576
    assert declare(plain, heuristic="torch").workspace_bytes == 8_519_680
    assert declare(heuristic="torch", workspace_bytes=4096).workspace_bytes == 4096  # torch with CUBLASLT_WORKSPACE_SIZE=4
    assert declare(heuristic="tilefish").heuristic == "tilefish"
    for bad in ("Torch", "cublas", 1):
        with pytest.raises(TilefishDeclarationError, match="heuristic must be one of \\['tilefish', 'torch'\\]"):
            declare(heuristic=bad)
    for bad in (1, "yes", 0):
        with pytest.raises(TilefishDeclarationError, match="allow_bf16_reduction must be a bool"):
            declare(allow_bf16_reduction=bad)
    assert declare(allow_bf16_reduction=True).allow_bf16_reduction is True
    with pytest.raises(TilefishDeclarationError, match="heuristic, allow_bf16_reduction and algos need an impl"):
        tilefish.library("x", heuristic="torch")
    with pytest.raises(TilefishDeclarationError, match="heuristic, allow_bf16_reduction and algos need an impl"):
        tilefish.library("x", algos=table())


def test_a_declaration_without_the_options_has_the_digest_and_the_entry_of_0_0_4():
    """The options are keys of the canonical declaration and fields of the entry only when they are not the default: an existing manifest and its
    pins stay valid, and what 0.0.4 reads is what it read."""
    op = declare()
    canonical = _library._canonical(op)
    assert set(canonical) == {"format", "op", "impl", "acc", "workspace_bytes", "params", "returns"} and canonical["format"] == 2
    entry = library_specs(op)[0].entry(SMS)
    assert not set(entry) & set(_library.OPTIONAL_ENTRY_FIELDS)
    # the heuristic named explicitly as the default is the default
    assert declaration_digest(declare(heuristic="tilefish", allow_bf16_reduction=False)) == declaration_digest(op)


def test_the_digest_covers_the_posture_the_opt_in_and_the_imported_algorithms_it_uses():
    base = declaration_digest(declare())
    torch = declaration_digest(declare(heuristic="torch", workspace_bytes=8_519_680))
    opt_in = declaration_digest(declare(allow_bf16_reduction=True))
    imported = declaration_digest(declare(algos=table()))
    assert len({base, torch, opt_in, imported}) == 4
    # a row the declaration does not use is not in its digest; a row it uses is, and so is the device of a table
    extra = {**table().algos, (32, 4096, 1024, True): SPLIT_BF16}
    assert declaration_digest(declare(algos=table(extra))) == imported
    changed = {**table().algos, (64, 2048, 1024, True): {**CONFIG, "tile_id": 81}}
    assert declaration_digest(declare(algos=table(changed))) != imported
    assert declaration_digest(declare(algos=table(sm_count=148))) != imported
    as_hex = {k: ALGO for k in table().algos}
    assert declaration_digest(declare(algos=table(as_hex))) != imported
    # two tables, in either order, are one declaration
    a, b = table(), table(sm_count=148)
    assert declaration_digest(declare(algos=[a, b])) == declaration_digest(declare(algos=[b, a]))


def test_an_entry_says_the_options_that_are_not_the_default():
    entry = library_specs(declare(heuristic="torch", allow_bf16_reduction=True, algos=table()))[1].entry(SMS)
    assert {k: entry[k] for k in _library.OPTIONAL_ENTRY_FIELDS} == {"heuristic": "torch", "algo_source": "imported", "allow_bf16_reduction": True}
    assert entry["workspace_bytes"] == 1_048_576
    only = library_specs(declare(allow_bf16_reduction=True))[0].entry(SMS)
    assert {k for k in only if k in _library.OPTIONAL_ENTRY_FIELDS} == {"allow_bf16_reduction"}


# -- the tables -----------------------------------------------------------------------------------------------------------------------------


def test_an_algorithm_table_is_checked_when_it_is_made():
    good = {(32, 2048, 1024, True): CONFIG}
    AlgoTable("sm_120", 170, VERSION, good)
    AlgoTable("sm_90", 132, VERSION, {(32, 2048, 1024, False): ALGO})  # a pin as 128 hex digits is taken as it is
    for args, message in (
        (("sm_103a", SMS, VERSION, good), "arch must look like 'sm_120'"),
        (("103", SMS, VERSION, good), "arch must look like"),
        (("sm_103", 0, VERSION, good), "sm_count must be a positive int"),
        (("sm_103", True, VERSION, good), "sm_count must be a positive int"),
        (("sm_103", SMS, "120804", good), "cublaslt_version must be a positive int"),
        (("sm_103", SMS, VERSION, {}), "algos must be a non-empty dict"),
        (("sm_103", SMS, VERSION, [CONFIG]), "algos must be a non-empty dict"),
        (("sm_103", SMS, VERSION, {(32, 2048, 1024): CONFIG}), "a key is \\(m, n, k, bias\\)"),
        (("sm_103", SMS, VERSION, {(32, 2048, 1024, 1): CONFIG}), "a key is \\(m, n, k, bias\\)"),
        (("sm_103", SMS, VERSION, {(0, 2048, 1024, True): CONFIG}), "a key is \\(m, n, k, bias\\)"),
        (("sm_103", SMS, VERSION, {(32, True, 1024, True): CONFIG}), "a key is \\(m, n, k, bias\\)"),
        (("sm_103", SMS, VERSION, {(32, 2048, 1024, True): {**CONFIG, "tile": 1}}), "has exactly the fields .*unknown \\['tile'\\]"),
        (("sm_103", SMS, VERSION, {(32, 2048, 1024, True): {k: v for k, v in CONFIG.items() if k != "id"}}), "missing \\['id'\\]"),
        (("sm_103", SMS, VERSION, {(32, 2048, 1024, True): {**CONFIG, "stages_id": -1}}), "stages_id must be an int from 0 to 4294967295"),
        (("sm_103", SMS, VERSION, {(32, 2048, 1024, True): {**CONFIG, "cluster_shape_id": 1 << 16}}), "cluster_shape_id must be an int from 0 to 65535"),
        (("sm_103", SMS, VERSION, {(32, 2048, 1024, True): {**CONFIG, "id": True}}), "id must be an int"),
        (("sm_103", SMS, VERSION, {(32, 2048, 1024, True): ALGO.upper()}), "128 lowercase hex characters"),
        (("sm_103", SMS, VERSION, {(32, 2048, 1024, True): 5}), "is a dict of"),
    ):
        with pytest.raises(TilefishDeclarationError, match=message):
            AlgoTable(*args)
    t = AlgoTable("sm_103", SMS, VERSION, good)
    assert t.device == ("sm_103", SMS, VERSION) and t.algos == good and t.algos[(32, 2048, 1024, True)] is not CONFIG  # a copy
    with pytest.raises(Exception):
        t.arch = "sm_90"  # frozen


def test_a_declaration_has_one_table_per_device_and_every_table_covers_every_entry():
    with pytest.raises(TilefishDeclarationError, match="algos must be an AlgoTable or a non-empty list of them"):
        declare(algos={(32, 2048, 1024, True): CONFIG})
    with pytest.raises(TilefishDeclarationError, match="algos must be an AlgoTable or a non-empty list of them"):
        declare(algos=[])
    with pytest.raises(TilefishDeclarationError, match="two algorithm tables are for sm_103 with 30 SMs and cuBLASLt 120804"):
        declare(algos=[table(), table()])
    assert declare(algos=[table(), table(sm_count=148)]).algos[1].sm_count == 148
    # the M64 entry needs a row in every table
    with pytest.raises(
        TilefishDeclarationError,
        match=r"the algorithm table for sm_103 with 30 SMs \(cuBLASLt 120804\) has no algorithm for \(m, n, k, bias\) = \(64, 2048, 1024, True\)",
    ):
        declare(algos=table({(32, 2048, 1024, True): CONFIG}))
    # a row of the other implementation does not count
    with pytest.raises(TilefishDeclarationError, match="has no algorithm for"):
        declare(plain, algos=table())


# -- the build ------------------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def device(monkeypatch):
    """A made-up sm_103 device with 30 SMs and cuBLASLt 120804, whose library calls are recorded."""
    calls = []

    class Pinned:
        algo_hex = ALGO

    monkeypatch.setattr(_library, "_device_index", lambda i: 0)
    monkeypatch.setattr(_library, "_capability", lambda i: 103)
    monkeypatch.setattr(_library, "sm_count", lambda i: SMS)
    monkeypatch.setattr(_cublaslt, "version", lambda: VERSION)
    monkeypatch.setattr(_cublaslt, "pin_algo", lambda *a, **kw: calls.append(("pin", a, kw)) or Pinned())
    monkeypatch.setattr(_cublaslt, "encode_algo", lambda cfg, **kw: calls.append(("encode", cfg)) or OTHER)
    monkeypatch.setattr(_cublaslt, "check_algo", lambda *a, **kw: calls.append(("check", a, kw)) or 0)
    return calls


def test_the_posture_and_the_opt_in_reach_the_heuristic(device):
    spec = library_specs(declare(heuristic="torch", allow_bf16_reduction=True))[0]
    got = build_library(spec, arch="sm_103")
    assert got.algo == ALGO
    assert device == [
        ("pin", (32, 2048, 1024, True), {"workspace_bytes": 1_048_576, "device_index": 0, "heuristic": "torch", "allow_bf16_reduction": True}),
        ("check", (32, 2048, 1024, True, ALGO), {"workspace_bytes": 1_048_576, "device_index": 0, "allow_bf16_reduction": True}),
    ]
    assert got.entry()["heuristic"] == "torch" and got.entry()["allow_bf16_reduction"] is True


def test_an_imported_algorithm_is_pinned_without_asking_the_heuristic(device):
    spec = library_specs(declare(algos=table()))[1]
    got = build_library(spec, arch="sm_103")
    # a configuration is encoded by the library, then checked as any pin is; the heuristic is not asked
    assert got.algo == OTHER and [c[0] for c in device] == ["encode", "check"] and device[0][1] == CONFIG
    assert device[1][1] == (64, 2048, 1024, True, OTHER)
    assert got.entry()["algo_source"] == "imported" and "heuristic" not in got.entry()
    # a pin given as hex is taken as it is
    device.clear()
    hexed = library_specs(declare(algos=table({k: ALGO for k in table().algos})))[0]
    assert build_library(hexed, arch="sm_103").algo == ALGO and [c[0] for c in device] == ["check"]


def test_a_device_no_table_is_for_is_refused_and_the_message_names_the_tables(device, monkeypatch):
    for patch, what in (
        (lambda: monkeypatch.setattr(_library, "sm_count", lambda i: 148), "sm_103 with 148 SMs and cuBLASLt 120804"),
        (lambda: monkeypatch.setattr(_cublaslt, "version", lambda: 130000), "sm_103 with 148 SMs and cuBLASLt 130000"),
    ):
        patch()
        spec = library_specs(declare(algos=[table(), table(arch="sm_120", sm_count=170)]))[0]
        with pytest.raises(TilefishBuildError, match=rf"cannot pin 'g.M32' on {what}: its declaration imports its algorithms \(algos=\), and has a "
                           r"table only for sm_103 with 30 SMs and cuBLASLt 120804, sm_120 with 170 SMs and cuBLASLt 120804"):  # fmt: skip
            build_library(spec, arch="sm_103")
    assert not [c for c in device if c[0] == "pin"]  # never the heuristic instead


def test_an_algorithm_the_checks_of_a_run_refuse_is_not_pinned(device, monkeypatch):
    def refuse(*a, **kw):
        raise tilefish.TilefishRuntimeError("reduction scheme 4 adds the split-K partial sums in bf16")

    monkeypatch.setattr(_cublaslt, "check_algo", refuse)
    with pytest.raises(TilefishBuildError, match="cannot pin 'g.M32' on sm_103: the imported algorithm is refused: reduction scheme 4"):
        build_library(library_specs(declare(algos=table()))[0], arch="sm_103")
    with pytest.raises(TilefishBuildError, match="the heuristic's algorithm is refused"):
        build_library(library_specs(declare())[0], arch="sm_103")


# -- the manifest and the replay ------------------------------------------------------------------------------------------------------------


def test_a_replay_takes_the_imported_algorithm_of_its_device_and_refuses_another(tmp_path, monkeypatch):
    monkeypatch.setattr(_cublaslt, "version", lambda: VERSION)
    monkeypatch.setattr(_cublaslt, "encode_algo", lambda cfg, **kw: OTHER)
    op = declare(algos=table())
    spec = library_specs(op)[0]
    path = tmp_path / "m" / "manifest.json"
    write_manifest([], path, cubins=True, library=[BuiltLibrary(spec, "sm_103", OTHER, VERSION, SMS)])
    m = Manifest.load(tmp_path / "m")
    assert m.library_entry("g.M32", "sm_103")["algo_source"] == "imported"
    assert built_from_manifest(m, spec, "sm_103", SMS, 0).algo == OTHER
    # the same declaration, another algorithm in the manifest (its digest does not say which algorithm the table gives on a device)
    write_manifest([], path, cubins=True, library=[BuiltLibrary(spec, "sm_103", ALGO, VERSION, SMS)], merge=True, overwrite=True)
    with pytest.raises(TilefishManifestError, match="has the algorithm 0123456789abcdef..., but the declaration imports fedcba9876543210... for this device"):
        built_from_manifest(Manifest.load(tmp_path / "m"), spec, "sm_103", SMS, 0)
    # an entry of the heuristic for a declaration that imports, and the reverse: the entries differ
    write_manifest([], path, cubins=True, library=[BuiltLibrary(library_specs(declare())[0], "sm_103", ALGO, VERSION, SMS)], merge=True, overwrite=True)
    with pytest.raises(TilefishManifestError, match="first difference at"):
        built_from_manifest(Manifest.load(tmp_path / "m"), spec, "sm_103", SMS, 0)
    with pytest.raises(TilefishManifestError, match="first difference at"):
        built_from_manifest(Manifest.load(tmp_path / "m"), library_specs(declare(heuristic="torch", workspace_bytes=8_519_680))[0], "sm_103", SMS, 0)


def entry_with(**fields):
    entry = BuiltLibrary(library_specs(declare())[0], "sm_103", ALGO, VERSION, SMS).entry()
    entry.update(fields)
    return entry


def test_the_optional_fields_are_absent_or_their_one_value():
    validate_entry(entry_with())
    validate_entry(entry_with(heuristic="torch", algo_source="imported", allow_bf16_reduction=True))
    for key, bad in (
        ("heuristic", "tilefish"),  # the default is written by leaving the field out
        ("heuristic", "Torch"),
        ("heuristic", None),
        ("algo_source", "heuristic"),
        ("algo_source", 1),
        ("allow_bf16_reduction", False),
        ("allow_bf16_reduction", 1),
        ("allow_bf16_reduction", "true"),
    ):
        with pytest.raises(TilefishManifestError, match=rf"{key} is absent \(the default\) or {json.dumps(_library.OPTIONAL_ENTRY_FIELDS[key])}, got"):
            validate_entry(entry_with(**{key: bad}))
    with pytest.raises(TilefishManifestError, match="may have \\['heuristic', 'algo_source', 'allow_bf16_reduction'\\]; unknown \\['reduction'\\]"):
        validate_entry(entry_with(reduction="bf16"))


def test_the_rust_emitter_says_how_the_algorithm_was_chosen(tmp_path):
    from tilefish.hosts import rust

    op = declare(heuristic="torch", allow_bf16_reduction=True, algos=table())
    path = tmp_path / "m" / "manifest.json"
    write_manifest([], path, cubins=True, library=[BuiltLibrary(library_specs(op)[0], "sm_103", ALGO, VERSION, SMS)])
    m = Manifest.load(tmp_path / "m")
    source = rust.render(m.check_pin(m.digest))
    assert "the one the declaration imported for the device (`algos=`)" in source
    assert "split-K to add its partial sums in bf16 (`allow_bf16_reduction`)" in source
