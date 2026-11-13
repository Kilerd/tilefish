"""One declaration, an implementation per architecture (``@tilefish.extern(by_arch=...)``), with no Triton and no GPU.

The fastest kernel that gives the bits a model needs can differ by architecture (the same ``tl.dot`` is lowered to other instructions on other
GPUs, which may sum in another order). An op declared with ``by_arch={"sm_103": tilefish.impl(...), ...}`` (and ``fallback=``) is one typed
signature for a graph, a plan and a caller, and an implementation per architecture: ``export``, ``jit`` and ``replay`` pick the one of the
target's architecture and refuse one with none. Every implementation has the stub's parameters and the same ABI; what may differ is the kernel,
its compile options, its constexpr values and its grid. Each implementation is an ``ExternOp`` of its own whose declaration digest, compile
specs and manifest entries carry its key (``"implementation": "sm_103"`` or ``"fallback"``), so a replay refuses an entry of another
implementation, and a changed implementation is stale.

``fake_compile`` stands in for Triton's frontend and compiler; ``test_gpu_per_arch.py`` runs real kernels.
"""

import importlib
import json
import linecache
import sys

import fake_compile
import pytest
from fake_jit import FakeJit
from fake_torch import FakeTensor

import tilefish
from tilefish import (
    Dim,
    Manifest,
    Out,
    Tensor,
    TilefishBuildError,
    TilefishDeclarationError,
    TilefishManifestError,
    TilefishRuntimeError,
    TilefishSpecError,
    const,
    f32,
    runtime,
)
from tilefish.aot import CompileSpec, build, compile_specs, declaration_digest

R = Dim("R")
MOD = "pa_kernels"


def wide(**kw):
    consts = {"ROWS": 4, **kw.pop("consts", {})}
    return tilefish.impl(triton=f"{MOD}:row_sum_wide", num_warps=8, consts=consts, **kw)


def narrow(**kw):
    return tilefish.impl(triton=f"{MOD}:row_sum_narrow", **kw)


def declare(by_arch=None, fallback=None, name="row_sum", **kw):
    """``y[r] = sum(x[r, :])``: two tilings of one row reduction, ``ROWS`` rows per program (the grid follows it)."""
    by_arch = {"sm_103": wide(), "sm_120": narrow()} if by_arch is None else by_arch

    @tilefish.extern(by_arch=by_arch, fallback=fallback, grid=lambda R, ROWS: (tilefish.cdiv(R, ROWS),), name=name, **kw)
    def row_sum(x: Tensor[R, 256, f32], y: Out[Tensor[R, f32]], n_rows: R, ROWS: const = 1, C: const = 256): ...

    return row_sum


# -- the declaration ----------------------------------------------------------------------------------------------------------------------------


def test_each_architecture_has_its_own_implementation_and_specs():
    op = declare(fallback=narrow(num_warps=2))
    assert set(op.implementations) == {"sm_103", "sm_120", "fallback"}
    a, b, c = op.for_arch(103), op.for_arch("sm_120"), op.for_arch("sm_90")
    assert (a.implementation, b.implementation, c.implementation) == ("sm_103", "sm_120", "fallback")
    assert a is op.implementations["sm_103"] and op.for_arch("sm_103") is a  # one object per implementation
    (sa,), (sb,), (sc,) = compile_specs(op, arch=103), compile_specs(op, arch="sm_120"), compile_specs(op, arch=90)
    assert sa.stem == sb.stem == sc.stem == "row_sum"  # one binary name: a plan names the same entry on every architecture
    assert (sa.triton, sa.num_warps, sa.grid, sa.constants) == (f"{MOD}:row_sum_wide", 8, "(cdiv(R, 4),)", {(3,): 4, (4,): 256})
    assert (sb.triton, sb.num_warps, sb.grid, sb.constants) == (f"{MOD}:row_sum_narrow", 4, "(R,)", {(3,): 1, (4,): 256})
    assert (sc.num_warps, sc.implementation) == (2, "fallback")
    assert sa.implementation == "sm_103" and sa.to_dict()["implementation"] == "sm_103"
    # the ABI is the op's: the same launch parameters, preconditions and dimensions
    assert sa.launch_params == sb.launch_params and sa.preconditions == sb.preconditions and sa.dims == sb.dims
    assert op.node_attrs() == {"by_arch": {"sm_103": f"{MOD}:row_sum_wide", "sm_120": f"{MOD}:row_sum_narrow", "fallback": f"{MOD}:row_sum_narrow"}}


def test_an_architecture_with_no_implementation_is_refused_and_specs_need_an_architecture():
    op = declare()
    with pytest.raises(TilefishRuntimeError, match=r"'row_sum' has no implementation for sm_90: it declares one for sm_103, sm_120 and no fallback"):
        op.for_arch(90)
    with pytest.raises(TilefishSpecError, match="no implementation for sm_100"):
        compile_specs(op, arch="sm_100")
    with pytest.raises(TilefishSpecError, match=r"has an implementation per architecture \(sm_103, sm_120\).*compile_specs\(op, arch=...\)"):
        compile_specs(op)
    with pytest.raises(TilefishSpecError, match=r"declaration_digest\(op.for_arch\(arch\)\)"):
        declaration_digest(op)
    plain = declare(by_arch={"sm_103": wide()}).for_arch(103)
    assert plain.for_arch(120) is plain  # an implementation (like any op with one) runs as it is
    assert compile_specs(plain, arch=120) == compile_specs(plain)


def test_the_identity_of_a_binary_is_its_implementation():
    op = declare(fallback=narrow())
    digests = {k: declaration_digest(v) for k, v in op.implementations.items()}
    assert len(set(digests.values())) == 3  # the fallback is the sm_120 implementation's kernel and options, under another key
    same = declare(fallback=narrow())
    assert {k: declaration_digest(v) for k, v in same.implementations.items()} == digests  # deterministic
    # another sm_120 implementation leaves the digest of the sm_103 one as it was: a manifest is per architecture
    other = declare(by_arch={"sm_103": wide(), "sm_120": narrow(num_warps=2)}, fallback=narrow())
    assert declaration_digest(other.for_arch(103)) == digests["sm_103"] and declaration_digest(other.for_arch(120)) != digests["sm_120"]
    # an op with one implementation is not any of them
    @tilefish.extern(triton=f"{MOD}:row_sum_wide", num_warps=8, grid=lambda R, ROWS: (tilefish.cdiv(R, ROWS),), name="row_sum")
    def row_sum(x: Tensor[R, 256, f32], y: Out[Tensor[R, f32]], n_rows: R, ROWS: const = 4, C: const = 256): ...

    (one,) = compile_specs(row_sum)
    (a,) = compile_specs(op, arch=103)
    assert one.implementation is None and "implementation" not in one.to_dict()
    da, do = a.to_dict(), one.to_dict()
    assert da.pop("implementation") == "sm_103" and da.pop("declaration_digest") != do.pop("declaration_digest")
    assert da == do  # the rest of the spec is that op's


def test_a_spec_round_trips_and_is_verified_against_its_implementation():
    op = declare(fallback=narrow())
    for arch in (103, 120, 90):
        (spec,) = compile_specs(op, arch=arch)
        back = CompileSpec.from_json(spec.to_json())
        assert back.implementation == spec.implementation and not back.verified
        back.verify_against(op)  # by its implementation key
        assert back.verified
        back.verify_against(op.for_arch(arch))
    doc = compile_specs(op, arch=103)[0].to_dict()
    doc["implementation"] = "sm_120"  # a self-consistent edit: the spec of the other implementation, by its key, is not this one
    with pytest.raises(TilefishSpecError, match="first difference at"):
        CompileSpec.from_dict(doc).verify_against(op)
    doc["implementation"] = "sm_90"
    with pytest.raises(TilefishSpecError, match=r"first difference at implementation: the spec has 'sm_90'"):
        CompileSpec.from_dict(doc).verify_against(op)
    for bad in ("sm_9", "Fallback", 103, None, ""):
        doc["implementation"] = bad
        with pytest.raises(TilefishSpecError, match="implementation"):
            CompileSpec.from_dict(doc)


def _wide_kernel(x, y, n_rows, ROWS, C):
    return x


def _swapped_kernel(x, y, ROWS, n_rows, C):
    return x


def kernel(fn, name):
    """A kernel object that lists its parameters (as a ``triton.jit`` function does), named ``pa_kernels:<name>``."""
    fn.__module__, fn.__name__ = MOD, name
    return FakeJit(fn)


def test_every_kernel_takes_the_stubs_parameters():
    ok = tilefish.impl(triton=kernel(_wide_kernel, "row_sum_wide"), num_warps=8)
    declare(by_arch={"sm_103": ok, "sm_120": narrow()})
    swapped = tilefish.impl(triton=kernel(_swapped_kernel, "row_sum_narrow"))
    with pytest.raises(TilefishDeclarationError, match=r"stub's parameters .* do not match the parameters of the Triton kernel 'pa_kernels:row_sum_narrow'"):
        declare(by_arch={"sm_103": ok, "sm_120": swapped})
    with pytest.raises(TilefishDeclarationError, match="do not match"):
        declare(by_arch={"sm_103": ok}, fallback=swapped)


def test_every_implementation_has_the_same_binaries_and_abi():
    Rb = Dim("Rb", buckets=[64, 128])
    with pytest.raises(TilefishDeclarationError, match=r"the implementation for sm_120 has the binaries \['row_sum.Rb64', 'row_sum.Rb128'\], the one for sm_103 \['row_sum'\]"):
        declare(by_arch={"sm_103": wide(), "sm_120": narrow(consts={"ROWS": Rb})})
    with pytest.raises(TilefishDeclarationError, match=r"sets \['BLOCK'\], which are not const parameters of the stub \(\['C', 'ROWS'\]\)"):
        declare(by_arch={"sm_103": wide(), "sm_120": narrow(consts={"BLOCK": 2})})
    with pytest.raises(TilefishDeclarationError, match="must default to a number"):
        declare(by_arch={"sm_103": wide(), "sm_120": narrow(consts={"ROWS": "two"})})
    with pytest.raises(TilefishDeclarationError, match=r"the implementation for the fallback: .*grid sizes must be at least 1"):
        declare(fallback=narrow(grid=lambda R: (0,)))
    # the same binaries, but what a host checks of a dimension differs: another ABI
    Rb16 = Dim("Rb", buckets=[64, 128], divisible_by=16)
    with pytest.raises(TilefishDeclarationError, match=r"the implementations for sm_103 and sm_120 are not launched the same way: 'row_sum.Rb64' differs at dims"):
        declare(by_arch={"sm_103": wide(consts={"ROWS": Rb}), "sm_120": narrow(consts={"ROWS": Rb16})})
    # a const that depends on a bucketed dimension in every implementation: the same binaries
    op = declare(by_arch={"sm_103": wide(consts={"ROWS": Rb}), "sm_120": narrow(consts={"ROWS": Rb})})
    assert [s.stem for s in compile_specs(op, arch=120)] == ["row_sum.Rb64", "row_sum.Rb128"]


def test_index_bits_is_the_ops_and_every_implementation_has_its_limits():
    op = declare(index_bits=64)
    for arch in (103, 120):
        (spec,) = compile_specs(op, arch=arch)
        assert not [p for p in spec.preconditions if p.kind == "max_elements"]
    (spec,) = compile_specs(declare(), arch=103)
    assert [p.arg for p in spec.preconditions if p.kind == "max_elements"] == ["x", "y"]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(triton="m:f"), r"triton= is not an argument of the op"),
        (dict(num_warps=8), r"num_warps= is not an argument of the op"),
        (dict(enable_fp_fusion=False), r"enable_fp_fusion= is not an argument of the op"),
        (dict(ptx="k.ptx", symbol="k"), r"ptx=, symbol= is not an argument of the op"),
        (dict(by_arch={}), r"by_arch= maps architectures to implementations"),
        (dict(by_arch=[("sm_103", "x")]), r"by_arch= maps architectures to implementations"),
        (dict(by_arch={"103": "x"}), r"keys are architectures such as 'sm_103'"),
        (dict(by_arch={"sm_090": "x"}), r"no leading zero"),
        (dict(by_arch={"sm_103": "m:f"}), r"by_arch\['sm_103'\] must be a tilefish.impl"),
        (dict(by_arch={"sm_103": None}), r"by_arch\['sm_103'\] must be a tilefish.impl"),
        (dict(fallback="m:f", by_arch=None), r"fallback= is the implementation of an op declared with by_arch="),
        (dict(fallback="m:f"), r"fallback= is a tilefish.impl"),
        (dict(index_bits=16), r"index_bits must be 32"),
        (dict(name="a-b"), r"name must be a plain identifier"),
    ],
)
def test_a_declaration_by_architecture_is_checked(kwargs, message):
    kwargs.setdefault("by_arch", {"sm_103": tilefish.impl(triton="m:f")})
    with pytest.raises(TilefishDeclarationError, match=message):
        tilefish.extern(grid=lambda R: (R,), **kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(), r"takes one kernel, triton= .* or gluon= .*, got none"),
        (dict(triton="m:f", gluon="m:g"), r"takes one kernel.*got triton and gluon"),
        (dict(triton="not a ref"), r"triton='not a ref' must look like 'module:function'"),
        (dict(triton="m:f", num_warps=3), r"num_warps must be a power of two"),
        (dict(triton="m:f", num_stages=0), r"num_stages must be a positive int"),
        (dict(triton="m:f", enable_fp_fusion=1), r"enable_fp_fusion must be True, False or None"),
        (dict(triton="m:f", consts=[("ROWS", 2)]), r"consts= maps const parameter names to values"),
    ],
)
def test_an_implementation_is_checked(kwargs, message):
    with pytest.raises(TilefishDeclarationError, match=message):
        tilefish.impl(**kwargs)


def test_an_implementation_may_be_a_gluon_kernel():
    op = declare(by_arch={"sm_103": tilefish.impl(gluon=f"{MOD}:row_sum_gluon"), "sm_120": narrow()})
    (a,), (b,) = compile_specs(op, arch=103), compile_specs(op, arch=120)
    assert (a.language, b.language) == ("gluon", "triton") and a.launch_params == b.launch_params
    assert a.to_dict()["language"] == "gluon" and a.triton == f"{MOD}:row_sum_gluon"


@tilefish.extern(by_arch={"sm_103": tilefish.impl(triton=f"{MOD}:row_sum_wide")}, grid=lambda R: (R,))
def pickled_sum(x: Tensor[R, 256, f32], y: Out[Tensor[R, f32]], n_rows: R, ROWS: const = 1, C: const = 256): ...


def test_an_implementation_pickles_by_its_op_and_its_key():
    import pickle

    one = pickled_sum.for_arch(103)
    assert pickle.loads(pickle.dumps(one)) is one and pickle.loads(pickle.dumps(pickled_sum)) is pickled_sum


def test_the_reference_is_the_ops_whatever_the_architecture():
    def reference(x):
        return x

    op = declare(reference=reference)
    assert op.reference is reference and all(one.reference is reference for one in op.implementations.values())


# -- export, manifest and replay (the fake frontend and compiler) ---------------------------------------------------------------------------

KERNELS = '''from fake_jit import gluon_jit, jit

SCALE = 1.0


@jit
def row_sum_wide(x, y, n_rows, ROWS, C):
    return (x * SCALE + n_rows, "wide", ROWS, C)


@jit
def row_sum_narrow(x, y, n_rows, ROWS, C):
    return (x * SCALE + n_rows, "narrow", ROWS, C)


@gluon_jit
def row_sum_gluon(x, y, n_rows, ROWS, C):
    return (x * SCALE + n_rows, "gluon", ROWS, C)
'''


@pytest.fixture
def world(tmp_path, monkeypatch):
    compiler = fake_compile.install(monkeypatch)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    directory = tmp_path / "project"
    directory.mkdir()
    (directory / f"{MOD}.py").write_text(KERNELS)
    monkeypatch.syspath_prepend(str(directory))
    sys.modules.pop(MOD, None)
    yield directory, compiler
    sys.modules.pop(MOD, None)


def load(directory, attr):
    """The kernel as a new process imports it from the file as it is now."""
    sys.modules.pop(MOD, None)
    importlib.invalidate_caches()
    linecache.clearcache()
    return getattr(importlib.import_module(MOD), attr)


def held(directory, spec, arch):
    """The IR digest a replay on ``arch`` recomputes for ``spec`` from the kernel the process holds."""
    return runtime._ir_digest_of(spec, load(directory, spec.triton.split(":")[1]), arch, {})


def test_export_builds_the_implementation_of_the_target_and_the_entry_says_which(world, tmp_path):
    _, compiler = world
    op = declare(fallback=narrow(num_warps=2))
    out = tmp_path / "build"
    m103 = runtime.export(op, out, arch=103)
    (entry,) = m103.doc["kernels"]
    assert (entry["arch"], entry["implementation"], entry["num_warps"], entry["grid"]) == ("sm_103", "sm_103", 8, "(cdiv(R, 4),)")
    assert entry["declaration_digest"] == declaration_digest(op.for_arch(103))
    assert [c["value"] for c in entry["constexprs"]] == [4, 256]
    m = runtime.export(op, out, arch=120)  # merged: an entry per architecture, each of its own implementation
    m = runtime.export(op, out, arch=90)
    by_arch = {e["arch"]: e for e in m.doc["kernels"]}
    assert {a: (e["implementation"], e["num_warps"], e["grid"]) for a, e in by_arch.items()} == {
        "sm_103": ("sm_103", 8, "(cdiv(R, 4),)"),
        "sm_120": ("sm_120", 4, "(R,)"),
        "sm_90": ("fallback", 2, "(R,)"),
    }
    assert compiler.count == 3 and m.digest != m103.digest  # the implementation is in the entry, so in the digest of the manifest
    with pytest.raises(TilefishRuntimeError, match="no implementation for sm_100"):
        runtime.export(declare(), tmp_path / "none", arch=100)  # no fallback: nothing is built
    assert compiler.count == 3


def test_a_replay_recomputes_the_identity_of_the_implementation_of_its_architecture(world, tmp_path, monkeypatch):
    directory, _ = world
    op = declare()
    out = tmp_path / "build"
    runtime.export(op, out, arch=103)
    manifest = runtime.export(op, out, arch=120)
    for arch in (103, 120):
        (spec,) = compile_specs(op, arch=arch)
        built = manifest.built_kernel(spec, f"sm_{arch}", ir_digest=held(directory, spec, arch), ir_triton="3.5.1")
        assert built.spec.implementation == f"sm_{arch}"
    # replay_built: the device's architecture picks the implementation, which is checked against the entry of that architecture
    monkeypatch.setattr(runtime, "_torch", lambda: None)
    monkeypatch.setattr(runtime, "_capability", lambda torch, index: 120)
    target = runtime._target(op).at(120)
    (spec,) = target.compile_specs()
    assert runtime.replay_built(target, spec, manifest, False, 0).sha256_code == manifest.get("row_sum", "sm_120").sha256_code
    # the entry of another architecture, under this one's spec: another implementation
    (s103,) = compile_specs(op, arch=103)
    with pytest.raises(TilefishManifestError, match=r"was built from the implementation 'sm_120' .* now runs its implementation 'sm_103' on sm_120"):
        manifest.built_kernel(s103, "sm_120", pinned=True)


def test_a_changed_implementation_is_stale_and_the_other_architectures_are_not(world, tmp_path):
    directory, _ = world
    out = tmp_path / "build"
    runtime.export(declare(), out, arch=103)
    manifest = runtime.export(declare(), out, arch=120)
    # the sm_103 implementation's options change: its declaration digest, pinned or not
    changed = declare(by_arch={"sm_103": wide(num_stages=2), "sm_120": narrow()})
    (s103,) = compile_specs(changed, arch=103)
    for pinned in (False, True):
        with pytest.raises(TilefishManifestError, match="first difference at"):
            manifest.built_kernel(s103, "sm_103", ir_digest=held(directory, s103, 103), ir_triton="3.5.1", pinned=pinned)
    (s120,) = compile_specs(changed, arch=120)  # the sm_120 binary is the one the declaration still gives
    manifest.built_kernel(s120, "sm_120", ir_digest=held(directory, s120, 120), ir_triton="3.5.1")
    # the map sends sm_103 to another kernel
    swapped = declare(by_arch={"sm_103": narrow(num_warps=8, consts={"ROWS": 4}), "sm_120": narrow()})
    (s103,) = compile_specs(swapped, arch=103)
    with pytest.raises(TilefishManifestError, match="first difference at"):
        manifest.built_kernel(s103, "sm_103", pinned=True)
    # sm_103 now falls back
    fallen = declare(by_arch={"sm_120": narrow()}, fallback=wide())
    (s103,) = compile_specs(fallen, arch=103)
    with pytest.raises(TilefishManifestError, match=r"from the implementation 'sm_103' .* now runs its implementation 'fallback' on sm_103: its binary is stale"):
        manifest.built_kernel(s103, "sm_103", pinned=True)
    # the kernel of the sm_103 implementation is edited: its IR, pinned or not
    (s103,) = compile_specs(declare(), arch=103)
    before = held(directory, s103, 103)
    (directory / f"{MOD}.py").write_text(KERNELS.replace('"wide"', '"wider"'))
    after = held(directory, s103, 103)
    assert after != before
    for pinned in (False, True):
        with pytest.raises(TilefishManifestError, match=r"built from Triton IR with digest .* now compiles to IR with digest .*stale"):
            manifest.built_kernel(s103, "sm_103", ir_digest=after, ir_triton="3.5.1", pinned=pinned)
    (s120,) = compile_specs(declare(), arch=120)
    manifest.built_kernel(s120, "sm_120", ir_digest=held(directory, s120, 120), ir_triton="3.5.1")  # the other kernel did not change


def test_an_op_with_one_implementation_and_an_entry_of_one_by_architecture_are_not_each_other(world, tmp_path):
    out = tmp_path / "build"
    manifest = runtime.export(declare(), out, arch=103)

    @tilefish.extern(triton=f"{MOD}:row_sum_wide", num_warps=8, grid=lambda R, ROWS: (tilefish.cdiv(R, ROWS),), name="row_sum")
    def row_sum(x: Tensor[R, 256, f32], y: Out[Tensor[R, f32]], n_rows: R, ROWS: const = 4, C: const = 256): ...

    (one,) = compile_specs(row_sum)
    with pytest.raises(TilefishManifestError, match=r"from the implementation 'sm_103' of an op declared with by_arch=, but the declaration now runs one implementation"):
        manifest.built_kernel(one, "sm_103", pinned=True)
    plain = runtime.export(row_sum, tmp_path / "plain", arch=103)
    (spec,) = compile_specs(declare(), arch=103)
    with pytest.raises(TilefishManifestError, match=r"from an op with one implementation, but the declaration now runs its implementation 'sm_103'"):
        plain.built_kernel(spec, "sm_103", pinned=True)
    with pytest.raises(TilefishManifestError, match="different binary|declaration_digest|replace"):
        runtime.export(row_sum, out, arch=103)  # the merge sees another binary under the same stem and architecture


def test_a_gluon_implementation_is_built_by_gluons_frontend_on_its_architecture(world, tmp_path):
    _, compiler = world
    op = declare(by_arch={"sm_103": tilefish.impl(gluon=f"{MOD}:row_sum_gluon"), "sm_120": narrow()})
    out = tmp_path / "build"
    runtime.export(op, out, arch=103)
    manifest = runtime.export(op, out, arch=120)
    kinds = {e["arch"]: (e["implementation"], e["source_digest_kind"]) for e in manifest.doc["kernels"]}
    assert kinds == {"sm_103": ("sm_103", "tilefish.gluon-ir.v1"), "sm_120": ("sm_120", "tilefish.triton-ir.v1")}
    assert compiler.languages == ["gluon", "triton"]


def test_the_implementation_of_one_architecture_is_not_built_for_another(world):
    (spec,) = compile_specs(declare(fallback=narrow()), arch=103)
    with pytest.raises(TilefishBuildError, match=r"'row_sum' is the implementation declared for sm_103, not for sm_120"):
        build(spec, arch=120)
    (fb,) = compile_specs(declare(fallback=narrow()), arch=90)
    assert build(fb, arch=90).spec.implementation == "fallback"


def edit_entry(path, change):
    doc = json.loads(path.read_text())
    change(doc["kernels"][0])
    path.write_text(json.dumps(doc))


@pytest.mark.parametrize("value", ["sm_120", "sm_90", "fallback2", "", 103, True, ["sm_103"]])
def test_the_reader_refuses_an_implementation_that_is_not_the_entrys(world, tmp_path, value):
    out = tmp_path / "build"
    runtime.export(declare(), out, arch=103)
    edit_entry(out / "manifest.json", lambda e: e.__setitem__("implementation", value))
    with pytest.raises(TilefishManifestError, match=r"implementation must be the entry's architecture 'sm_103' or 'fallback'"):
        Manifest.load(out)


def test_the_reader_takes_null_as_absent_and_fallback_on_any_architecture(world, tmp_path):
    out = tmp_path / "build"
    runtime.export(declare(), out, arch=103)
    edit_entry(out / "manifest.json", lambda e: e.__setitem__("implementation", "fallback"))
    assert Manifest.load(out).get("row_sum", "sm_103").entry["implementation"] == "fallback"
    edit_entry(out / "manifest.json", lambda e: e.__setitem__("implementation", None))
    Manifest.load(out)


def test_the_jit_and_replay_of_a_call_take_the_implementation_of_the_devices_architecture(monkeypatch):
    import fake_torch

    op = declare()
    x = FakeTensor((10, 256), "f32", 1 << 40)
    for cap, rows, grid in ((103, 4, (3,)), (120, 1, (10,))):
        monkeypatch.setattr(runtime, "_capability", lambda torch, index, cap=cap: cap)
        monkeypatch.setattr(runtime, "_device_index", lambda torch, device: 0)
        prepared, error = fake_torch.prepare(op, [x], next_slab=1 << 44)
        assert error is None, error
        assert prepared.spec.implementation == f"sm_{cap}" and prepared.grid == grid
        assert prepared.target.launch_extern() is op.for_arch(cap) and dict(prepared.spec.constants)[(3,)] == rows
    monkeypatch.setattr(runtime, "_capability", lambda torch, index: 90)
    prepared, error = fake_torch.prepare(op, [x], next_slab=1 << 44)
    assert isinstance(error, TilefishRuntimeError) and "no implementation for sm_90" in str(error)


# -- plans ---------------------------------------------------------------------------------------------------------------------------------


def block(op):
    Mb = Dim("Mb", buckets=[8, 16])

    @tilefish.extern(triton=f"{MOD}:scale", grid=lambda Mb: (Mb,), name="scale")
    def scale(x: Tensor[Mb, 256, f32], y: Out[Tensor[Mb, 256, f32]], n: Mb): ...

    @tilefish.graph
    def summed(x: Tensor[Mb, 256, f32]) -> Tensor[Mb, f32]:
        return op(scale(x))

    return summed


def test_a_plan_is_one_document_for_every_implementation():
    op = declare(fallback=narrow())
    g = block(op)
    p = tilefish.plan(g)
    assert [st["entry"] for st in p.steps(Mb=8)] == ["scale", "row_sum"]
    assert p.callees["row_sum"].implementations is not None and set(p.callees["row_sum"].implementations) == {"sm_103", "sm_120", "fallback"}
    # the same graph over an op with only the sm_103 implementation makes the same plan: a plan says nothing of the architecture
    assert tilefish.plan(block(declare(by_arch={"sm_103": wide()}))).digest == p.digest


def test_a_plan_runs_the_implementation_of_its_devices_architecture():
    from test_planrun import FakeBackend, FakeProvider

    from tilefish._planrun import PlanRunner

    class Device(FakeBackend):
        def __init__(self, cap):
            super().__init__()
            self.cap = cap

        def capability(self, index):
            return self.cap

    p = tilefish.plan(block(declare()))
    x = FakeTensor((16, 256), "f32", 1 << 36)
    grids = {}
    for cap in (103, 120):
        be = Device(cap)
        PlanRunner(p, FakeProvider(), be).run((x,), {})
        grids[cap] = [(e[1], e[2]) for e in be.launches()]
    assert grids == {103: [("scale", (16,)), ("row_sum", (4,))], 120: [("scale", (16,)), ("row_sum", (16,))]}
    with pytest.raises(TilefishRuntimeError, match="no implementation for sm_90"):
        PlanRunner(p, FakeProvider(), Device(90)).run((x,), {})


SCALE_KERNEL = '''

@jit
def scale(x, y, n):
    return (x * 2.0, n)
'''


def test_the_export_of_a_plan_and_of_a_plan_set_builds_the_implementation_of_the_target(world, tmp_path):
    directory, compiler = world
    (directory / f"{MOD}.py").write_text(KERNELS + SCALE_KERNEL)
    op = declare()
    g = block(op)
    p103 = runtime.export(g, tmp_path / "plan", arch=103)
    p120 = runtime.export(g, tmp_path / "plan120", arch=120)
    for p, arch, ws in ((p103, "sm_103", 8), (p120, "sm_120", 4)):
        manifest = Manifest.load(tmp_path / ("plan" if arch == "sm_103" else "plan120"))
        entry = manifest.get("row_sum", arch).entry
        assert (p.arch, entry["implementation"], entry["num_warps"]) == (arch, arch, ws)
    assert p103.stale_reason(p120.doc) is None  # the same plan, other manifests
    with pytest.raises(TilefishRuntimeError, match="no implementation for sm_90"):
        runtime.export(g, tmp_path / "plan90", arch=90)

    @tilefish.graph
    def other(x: Tensor[Dim("Mb", buckets=[8, 16]), 256, f32]) -> Tensor[Dim("Mb", buckets=[8, 16]), f32]:
        return op(x)

    s = runtime.export([g, other], tmp_path / "set", arch=120)
    entries = [e for e in Manifest.load(tmp_path / "set").doc["kernels"] if e["stem"] == "row_sum"]
    assert [(e["arch"], e["implementation"]) for e in entries] == [("sm_120", "sm_120")]  # built once for both plans
    assert s.manifest_digest == Manifest.load(tmp_path / "set").digest


def test_the_typed_rust_launchers_describe_each_architectures_implementation(world, tmp_path):
    from tilefish.hosts import rust

    op = declare()
    out = tmp_path / "build"
    runtime.export(op, out, arch=103)
    manifest = runtime.export(op, out, arch=120)
    text = rust.render(manifest, ops=[op])
    assert "pub struct RowSum" in text and "An implementation per architecture" in text
    assert "sm_103: the implementation `sm_103`, grid `(cdiv(R, 4),)`, block 256 threads" in text
    assert "sm_120: the implementation `sm_120`, grid `(R,)`, block 128 threads" in text
    # the declaration is checked against each architecture's entry: an op whose sm_120 implementation changed is not what was built
    changed = declare(by_arch={"sm_103": wide(), "sm_120": narrow(num_warps=2)})
    with pytest.raises(TilefishManifestError, match=r"binary 'row_sum' was built from \(on sm_120\)"):
        rust.render(manifest, ops=[changed])
    with pytest.raises(TilefishManifestError, match="no implementation for sm_120"):
        rust.render(manifest, ops=[declare(by_arch={"sm_103": wide()})])
