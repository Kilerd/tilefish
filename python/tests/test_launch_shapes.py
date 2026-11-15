"""Launch shapes (manifest schema 3, sections 4.3 to 4.6), with no GPU: grids that multiply dimensions, persistent grids, a configuration per bucket
of an extern, thread-block clusters and a global scratch. The declarations, the compile specs and their digests, the build's checks (on the fake
compiler), the manifest and its strict reader, what a launch is given (a stand-in for a loaded kernel), and a plan on a fake backend. The launches on
a GPU are ``test_gpu_launch_shapes.py``.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import copy
import importlib
import json
import linecache
import math
import sys
from types import SimpleNamespace

import fake_compile
import pytest
from binary_fakes import write_ptx
from built_fakes import fake_build
from fake_torch import FakeTensor, prepare
from test_planrun import FakeBackend, FakeProvider

import tilefish
from tilefish import (
    Dim,
    InOut,
    Manifest,
    Out,
    Tensor,
    TilefishBuildError,
    TilefishDeclarationError,
    TilefishManifestError,
    TilefishRuntimeError,
    TilefishSpecError,
    TilefishTypeError,
    _plan,
    const,
    f32,
)
from tilefish import _cuda
from tilefish._build import LaunchAttributes, spec_entry
from tilefish._planrun import PlanRunner
from tilefish.aot import CompileSpec, build, compile_specs, declaration_digest, write_manifest

T, C = Dim("T"), Dim("C")
TB = Dim("TB", buckets=[64, 4096])
ARCH = 103


def scale_over(grid, **kw):
    @tilefish.extern(triton="lk:scale", num_warps=4, grid=grid, name="scale", **kw)
    def scale(x: Tensor[T, C, f32], y: Out[Tensor[T, C, f32]], n: const = 0, BLOCK: const = 1024): ...

    return scale


# -- grids that multiply dimensions ---------------------------------------------------------------------------------------------------------------


def test_a_grid_may_multiply_two_dimensions_and_a_host_computes_it():
    op = scale_over(lambda T, C, BLOCK: (tilefish.cdiv(T * C, BLOCK),))
    assert op.grid_string == "(cdiv(C*T, 1024),)"  # the factors in their canonical order
    (spec,) = compile_specs(op)
    assert spec.grid == "(cdiv(C*T, 1024),)" and spec.verified
    CompileSpec.from_dict(spec.to_dict()).verify_against(op)  # the grammar of a spec takes it
    # one binary for every T and C: the grid is computed at each launch from the shapes
    for t, c, programs in ((7, 300, 3), (1, 1, 1), (1024, 1024, 1024), (5, 2048, 10)):
        prepared, error = prepare(op, [FakeTensor((t, c), "f32", 1 << 40)], next_slab=1 << 44)
        assert error is None and prepared.grid == (programs,), (t, c, error)
    # the product is in the declaration digest, as a product (not the affine grid of the same text without it)
    assert declaration_digest(op) != declaration_digest(scale_over(lambda T, C, BLOCK: (tilefish.cdiv(T + C, BLOCK),)))
    assert declaration_digest(op) == declaration_digest(scale_over(lambda T, C, BLOCK: (tilefish.cdiv(C * T, BLOCK),)))
    # a coefficient, a sum and several entries: what the text is, the value of a launch
    two = scale_over(lambda T, C: (2 * T * C + 1, T, C * C))
    assert two.grid_string == "(2*C*T + 1, T, C*C)"
    prepared, error = prepare(two, [FakeTensor((3, 5), "f32", 1 << 40)], next_slab=1 << 44)
    assert error is None and prepared.grid == (31, 3, 25)


def test_a_product_stays_refused_where_a_host_solves_it():
    with pytest.raises(TilefishTypeError, match="cannot multiply dimensions"):
        T * C  # noqa: B018 - outside a grid the algebra is affine
    with pytest.raises(TilefishTypeError, match="cannot multiply dimensions"):

        @tilefish.extern(triton="lk:scale", num_warps=4, grid=lambda T: (T,))
        def flat(x: Tensor[T * C, f32], y: Out[Tensor[T, f32]]): ...  # a shape: a host binds the dimensions of a call by solving it


def test_the_readers_take_a_grid_that_multiplies(tmp_path):
    op = scale_over(lambda T, C, BLOCK: (tilefish.cdiv(T * C, BLOCK),))
    (spec,) = compile_specs(op)
    write_manifest([fake_build(spec, arch=f"sm_{ARCH}")], tmp_path / "manifest.json", cubins=True)
    entry = Manifest.load(tmp_path).built_kernel(spec, f"sm_{ARCH}", pinned=True)
    assert entry.spec.grid == "(cdiv(C*T, 1024),)"


# -- a configuration per bucket ---------------------------------------------------------------------------------------------------------------------


def rows(by_bucket=None, **kw):
    @tilefish.extern(triton="lk:rows", num_warps=4, grid=lambda TB, BLOCK_T: (tilefish.cdiv(TB, BLOCK_T),), by_bucket=by_bucket, name="rows", **kw)
    def rows(x: Tensor[TB, 128, f32], y: Out[Tensor[TB, 128, f32]], n_rows: TB, BLOCK_T: const = 16, EPS: const[f32] = 1e-6): ...

    return rows


SMALL = tilefish.config(num_warps=2, BLOCK_T=32)
BIG = tilefish.config(num_warps=8, num_stages=2, BLOCK_T=128)


def test_each_bucket_is_built_with_its_own_configuration():
    op = rows(lambda TB: {64: SMALL, 4096: BIG}[TB])
    small, big = compile_specs(op)
    assert (small.stem, big.stem) == ("rows.TB64", "rows.TB4096")  # one binary per bucket, as a bucketed constexpr makes
    assert (small.num_warps, small.num_stages, small.block) == (2, 3, (64, 1, 1))
    assert (big.num_warps, big.num_stages, big.block) == (8, 2, (256, 1, 1))
    consts = lambda s: {a.name: a.value for a in s.args if a.kind == "constexpr"}  # noqa: E731
    assert consts(small) == {"BLOCK_T": 32, "EPS": 1e-6} and consts(big) == {"BLOCK_T": 128, "EPS": 1e-6}
    assert (small.grid, big.grid) == ("(2,)", "(32,)")  # the grid under each bucket's BLOCK_T
    assert small.ast_source_inputs().options == {"num_warps": 2, "num_stages": 3} and big.ast_source_inputs().options == {"num_warps": 8, "num_stages": 2}
    assert small.declaration_digest == big.declaration_digest  # one declaration, which covers every bucket's configuration
    for s in (small, big):
        CompileSpec.from_dict(s.to_dict()).verify_against(op)
    # a launch at a bucket takes that bucket's grid
    prepared, error = prepare(op, [FakeTensor((4096, 128), "f32", 1 << 40)], next_slab=1 << 44)
    assert error is None and prepared.spec.stem == "rows.TB4096" and prepared.grid == (32,)


def test_the_declaration_digest_covers_the_configuration_of_every_bucket():
    base = declaration_digest(rows(lambda TB: {64: SMALL, 4096: BIG}[TB]))
    assert base != declaration_digest(rows())  # (an op without by_bucket= digests as it did: no key)
    assert base != declaration_digest(rows(lambda TB: {64: SMALL, 4096: tilefish.config(num_warps=8, num_stages=2, BLOCK_T=64)}[TB]))
    assert base != declaration_digest(rows(lambda TB: {64: tilefish.config(num_warps=1, BLOCK_T=32), 4096: BIG}[TB]))
    assert base == declaration_digest(rows(lambda TB: SMALL if TB < 100 else BIG))  # what a configuration is, not how it is written
    # a bucket the callable leaves to the declaration (None) is the declaration's
    op = rows(lambda TB: None if TB == 64 else BIG)
    small, _ = compile_specs(op)
    assert (small.num_warps, small.grid) == (4, "(4,)")


def test_a_configuration_is_refused_where_it_cannot_be_one(tmp_path):
    for by_bucket, text in (
        (lambda TB: tilefish.config(BLOCK=64), "not a const parameter"),
        (lambda TB: 4, "returns a tilefish.config"),
        (lambda X: None, "not a bucketed dimension"),
        (lambda: None, "names no bucketed dimension"),
        (lambda TB: tilefish.config(EPS="a"), "must default to a number"),
        (64, "is a callable"),
    ):
        with pytest.raises(TilefishDeclarationError, match=text):
            rows(by_bucket)
    for bad, text in ((dict(num_warps=3), "num_warps"), (dict(num_stages=0), "num_stages"), (dict(num_ctas=3), "num_ctas")):
        with pytest.raises(TilefishDeclarationError, match=text):
            tilefish.config(**bad)
    with pytest.raises(TilefishDeclarationError, match="prebuilt binary"):

        @tilefish.extern(ptx=str(write_ptx(tmp_path)), symbol="axpy", block=(64,), grid=lambda TB: (TB,), by_bucket=lambda TB: None)
        def binary(x: Tensor[TB, f32], y: Out[Tensor[TB, f32]]): ...


def test_a_call_does_not_give_a_constexpr_a_bucket_chooses():
    op = rows(lambda TB: {64: SMALL, 4096: BIG}[TB])
    p = next(p for p in op.signature.consts if p.name == "BLOCK_T")
    with pytest.raises(TilefishTypeError, match="by_bucket= chooses"):
        op.check_const(p, 16)


def test_a_configuration_per_bucket_with_a_bucketed_constexpr_and_an_implementation_per_architecture():
    D = Dim("D", buckets=[64, 128])

    @tilefish.extern(
        triton="lk:mix", num_warps=4, grid=lambda TB, BLOCK_T: (tilefish.cdiv(TB, BLOCK_T),), by_bucket=lambda TB: SMALL if TB == 64 else BIG
    )
    def mix(x: Tensor[TB, D, f32], y: Out[Tensor[TB, D, f32]], BLOCK_T: const = 16, W: const = D): ...

    specs = compile_specs(mix)
    assert [s.stem for s in specs] == ["mix.D64.TB64", "mix.D64.TB4096", "mix.D128.TB64", "mix.D128.TB4096"]  # the bucketed constexpr first
    assert [s.num_warps for s in specs] == [2, 8, 2, 8]

    @tilefish.extern(
        by_arch={
            "sm_103": tilefish.impl(triton="lk:rows_a", num_warps=4, by_bucket=lambda TB: {64: SMALL, 4096: BIG}[TB]),
            "sm_120": tilefish.impl(triton="lk:rows_b", num_warps=2, by_bucket=lambda TB: None),
        },
        grid=lambda TB, BLOCK_T: (tilefish.cdiv(TB, BLOCK_T),),
    )
    def per_arch(x: Tensor[TB, 128, f32], y: Out[Tensor[TB, 128, f32]], BLOCK_T: const = 16): ...

    a = compile_specs(per_arch, arch=103)
    b = compile_specs(per_arch, arch=120)
    assert [s.num_warps for s in a] == [2, 8] and [s.num_warps for s in b] == [2, 2]


def test_a_configuration_per_bucket_in_a_plan_names_each_buckets_binary(tmp_path):
    op = rows(lambda TB: {64: SMALL, 4096: BIG}[TB])

    @tilefish.graph
    def g(x: Tensor[TB, 128, f32]):
        return op(x)

    plan = tilefish.plan(g)
    steps = {b["values"]["TB"]: b["steps"][0]["entry"] for b in plan.doc["buckets"]}
    assert steps == {64: "rows.TB64", 4096: "rows.TB4096"}


# -- persistent grids -------------------------------------------------------------------------------------------------------------------------------


def tiles(persistent=None, grid=None, **kw):
    @tilefish.extern(
        triton="lk:tiles", num_warps=4, grid=grid or (lambda T, C: (tilefish.cdiv(T, 128) * tilefish.cdiv(C, 128),)), persistent=persistent, name="tiles",
        **kw,
    )  # fmt: skip
    def tiles(x: Tensor[T, C, f32], y: Out[Tensor[T, C, f32]], n_t: T, n_c: C): ...

    return tiles


def stand_in(built, sms=None, limits=(2**31 - 1, 65535, 65535)):
    """A loaded kernel without a driver: what a launch is given (its grid of blocks, its scratch) is decided by these."""
    k = object.__new__(_cuda.LoadedKernel)
    k.built = built
    k._abi = built.abi()["params"]
    k.grid_limits = limits
    k.block = built.block
    k.persistent = built.spec.persistent
    k.sms = sms
    return k


def test_a_persistent_grid_is_the_tiles_and_the_host_launches_at_most_programs_per_sm_per_sm():
    op = tiles(tilefish.persistent())
    (spec,) = compile_specs(op)
    assert spec.persistent == {"programs_per_sm": 1} and spec.grid == "(cdiv(C, 128)*cdiv(T, 128),)"
    assert spec_entry(spec)["persistent"] == {"programs_per_sm": 1}
    built = fake_build(spec, arch=f"sm_{ARCH}")
    for sms, grid, programs in ((30, (100,), 30), (30, (12,), 12), (148, (100,), 100), (1, (5,), 1)):
        k = stand_in(built, sms)
        assert k.programs(grid) == (programs,) and k.launch_grid(grid) == (programs, 1, 1), (sms, grid)
    two = fake_build(compile_specs(tiles(tilefish.persistent(programs_per_sm=2)))[0], arch=f"sm_{ARCH}")
    assert stand_in(two, 30).programs((100,)) == (60,)
    assert _cuda.persistent_programs(100, 1, 30, 2) == 15 and _cuda.persistent_programs(100, 1, 1, 2) == 1  # a cluster of 2 CTAs per program, and at least one
    # the declaration digest and the spec have it (only when it is declared: every other digest is as it was)
    assert declaration_digest(op) != declaration_digest(tiles())
    assert declaration_digest(op) != declaration_digest(tiles(tilefish.persistent(programs_per_sm=2)))
    assert "persistent" not in compile_specs(tiles())[0].to_dict()
    CompileSpec.from_dict(spec.to_dict()).verify_against(op)


def test_a_persistent_kernel_whose_bytes_depend_on_the_sm_count_runs_on_that_count_only():
    (spec,) = compile_specs(tiles(tilefish.persistent(sm_count=30)))
    assert spec.persistent == {"programs_per_sm": 1, "sm_count": 30}
    built = fake_build(spec, arch=f"sm_{ARCH}")
    _cuda.check_sm_count(built, 30)
    with pytest.raises(TilefishRuntimeError, match="declared for 30 SMs .* has 148"):
        _cuda.check_sm_count(built, 148)
    with _cuda.sm_partition(30):  # a partition of 30 SMs of a larger device (a green context): the host says which
        assert _cuda.sm_count(0) == 30 and _cuda.current_partition() == 30
    assert _cuda.current_partition() is None


def test_a_persistent_grid_is_refused_where_it_cannot_be_one():
    with pytest.raises(TilefishDeclarationError, match="a persistent grid has one entry"):
        tiles(tilefish.persistent(), grid=lambda T, C: (T, C))
    for bad in (dict(programs_per_sm=0), dict(programs_per_sm=33), dict(programs_per_sm=True), dict(sm_count=0)):
        with pytest.raises(TilefishDeclarationError, match="tilefish.persistent"):
            tilefish.persistent(**bad)
    with pytest.raises(TilefishDeclarationError, match="persistent= is a tilefish.persistent"):
        tiles(persistent=1)
    (spec,) = compile_specs(tiles(tilefish.persistent()))
    for change, text in (
        ({"persistent": {"programs_per_sm": 0}}, "programs_per_sm"),
        ({"persistent": {"programs_per_sm": 1, "extra": 1}}, "unknown"),
        ({"persistent": {"programs_per_sm": 1, "sm_count": 0}}, "sm_count"),
        ({"persistent": {"programs_per_sm": 1}, "grid": "(T, C)"}, "one entry"),
    ):
        d = {**spec.to_dict(), **change}
        with pytest.raises(TilefishSpecError, match=text):
            CompileSpec.from_dict(d)


def test_the_reader_holds_a_persistent_entry_to_its_rules(tmp_path):
    (spec,) = compile_specs(tiles(tilefish.persistent()))
    write_manifest([fake_build(spec, arch=f"sm_{ARCH}")], tmp_path / "manifest.json", cubins=True)
    assert Manifest.load(tmp_path).built_kernel(spec, f"sm_{ARCH}", pinned=True).spec.persistent == {"programs_per_sm": 1}
    path = tmp_path / "manifest.json"
    good = json.loads(path.read_text())
    for change, text in (
        (lambda e: e.update(persistent={"programs_per_sm": 40}), "programs_per_sm"),
        (lambda e: e.update(persistent={"programs_per_sm": 1, "sm_count": 1.5}), "sm_count"),
        (lambda e: e.update(persistent={"programs_per_sm": 1}, grid="(T, C)"), "one entry"),
        (lambda e: e.update(persistent=[1]), "persistent must be an object"),
    ):
        doc = copy.deepcopy(good)
        change(doc["kernels"][0])
        path.write_text(json.dumps(doc))
        with pytest.raises(TilefishManifestError, match=text):
            Manifest.load(tmp_path)
    doc = copy.deepcopy(good)
    del doc["kernels"][0]["persistent"]  # the entry of another declaration: the spec says persistent, the entry does not
    path.write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match="not what the compile spec asks for"):
        Manifest.load(tmp_path).built_kernel(spec, f"sm_{ARCH}", pinned=True)
    # and the other way: an entry that says persistent of a declaration that does not (a host launches the entry's own grid)
    (plain,) = compile_specs(tiles())
    write_manifest([fake_build(plain, arch=f"sm_{ARCH}")], tmp_path / "plain" / "manifest.json", cubins=True)
    doc = json.loads((tmp_path / "plain" / "manifest.json").read_text())
    doc["kernels"][0]["persistent"] = {"programs_per_sm": 1}
    (tmp_path / "plain" / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match="the persistent grid"):
        Manifest.load(tmp_path / "plain").built_kernel(plain, f"sm_{ARCH}", pinned=True)


# -- thread-block clusters ------------------------------------------------------------------------------------------------------------------------------

KERNELS = '''from fake_jit import jit


@jit
def scale(x, y, n_elem, BLOCK):
    return (x, y, n_elem, BLOCK)


@jit
def desc(x, y, n_elem, BLOCK):
    return (x, y, n_elem, BLOCK)
'''


@pytest.fixture
def world(tmp_path, monkeypatch):
    compiler = fake_compile.install(monkeypatch)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    directory = tmp_path / "project"
    directory.mkdir()
    (directory / "lk.py").write_text(KERNELS)
    monkeypatch.syspath_prepend(str(directory))
    sys.modules.pop("lk", None)
    importlib.invalidate_caches()
    linecache.clearcache()
    monkeypatch.setenv("TILEFISH_CACHE_DIR", str(tmp_path / "cache"))
    yield directory, compiler
    sys.modules.pop("lk", None)


def flat(name="scale", **kw):
    n = Dim("n")

    @tilefish.extern(triton=f"lk:{name}", num_warps=4, grid=lambda n, BLOCK: (tilefish.cdiv(n, BLOCK),), name=name, **kw)
    def op(x: Tensor[n, f32], y: Out[Tensor[n, f32]], n_elem: n, BLOCK: const = 1024): ...

    return op


def test_a_cluster_of_ctas_is_compiled_recorded_and_launched_as_one(world, tmp_path):
    _, compiler = world
    op = flat(num_ctas=2)
    (spec,) = compile_specs(op)
    assert spec.num_ctas == 2 and spec.ast_source_inputs().options == {"num_warps": 4, "num_stages": 3, "num_ctas": 2}
    built = build(spec, arch=ARCH)
    assert compiler.calls[-1][1]["num_ctas"] == 2  # Triton compiles it with the option
    assert (built.launch.cluster_dims, built.launch.num_ctas) == ((2, 1, 1), 2) and built.launch.replay_blockers() == []
    write_manifest([built], tmp_path / "out" / "manifest.json", cubins=True)
    entry = json.loads((tmp_path / "out" / "manifest.json").read_text())["kernels"][0]
    assert entry["options"]["num_ctas"] == 2 and entry["launch"]["cluster_dims"] == [2, 1, 1] and entry["launch"]["num_ctas"] == 2
    replayed = Manifest.load(tmp_path / "out").built_kernel(spec, f"sm_{ARCH}", pinned=True)
    k = stand_in(replayed)
    assert k.launch_grid((5,)) == (10, 1, 1)  # a program is a cluster of 2 CTAs along x: 5 programs are 10 blocks
    # the digest has it only when it is not 1 (every other declaration digests as it did)
    assert declaration_digest(op) != declaration_digest(flat())
    assert "num_ctas" not in compile_specs(flat())[0].to_dict()


def test_a_cluster_that_is_not_the_declared_one_is_refused(world, tmp_path):
    _, compiler = world
    compiler.cluster_dims = (4, 1, 1)
    with pytest.raises(TilefishBuildError, match="cluster of .4, 1, 1. with num_ctas=2, but the declaration says num_ctas=2"):
        build(compile_specs(flat(num_ctas=2))[0], arch=ARCH)
    for bad in (3, 16, 0, True):
        with pytest.raises(TilefishDeclarationError, match="num_ctas"):
            flat(num_ctas=bad)
    with pytest.raises(TilefishDeclarationError, match="cluster= is the thread-block cluster of a prebuilt binary"):
        flat(cluster=(2, 1, 1))


def test_the_reader_holds_a_cluster_to_its_compile_options(world, tmp_path):
    (spec,) = compile_specs(flat(num_ctas=2))
    write_manifest([build(spec, arch=ARCH)], tmp_path / "manifest.json", cubins=True)
    path = tmp_path / "manifest.json"
    good = json.loads(path.read_text())
    for change in (
        lambda e: e["options"].pop("num_ctas"),
        lambda e: e["options"].update(num_ctas=4),
        lambda e: e["options"].update(num_ctas=1),
    ):
        doc = copy.deepcopy(good)
        change(doc["kernels"][0])
        path.write_text(json.dumps(doc))
        with pytest.raises(TilefishManifestError, match="options.num_ctas"):
            Manifest.load(tmp_path)


def test_a_prebuilt_binary_declares_its_cluster(tmp_path):
    ptx = write_ptx(tmp_path)
    n = Dim("n")

    def declare(cluster):
        @tilefish.extern(ptx=str(ptx), symbol="axpy", block=(128,), grid=lambda n: (tilefish.cdiv(n, 256),), cluster=cluster, name="axpy")
        def axpy(x: Tensor[n, f32], y: InOut[Tensor[n, f32]], n_elem: n): ...

        return axpy

    op = declare((2,))
    (spec,) = compile_specs(op)
    assert spec.binary["cluster"] == [2, 1, 1] and op.binary.cluster == (2, 1, 1)
    assert declaration_digest(op) != declaration_digest(declare(None)) and "cluster" not in compile_specs(declare(None))[0].binary
    from binary_fakes import fake_binary_build

    built = fake_binary_build(spec, op.binary, arch=f"sm_{ARCH}")
    assert (built.launch.cluster_dims, built.launch.num_ctas) == ((2, 1, 1), 2)
    assert stand_in(built).launch_grid((3,)) == (6, 1, 1)  # its grid counts clusters
    for bad in ((3, 3), (0,), (9,), ("a",)):
        with pytest.raises(TilefishDeclarationError, match="cluster"):
            declare(bad)
    with pytest.raises(TilefishDeclarationError, match="num_ctas= is a Triton compile option"):

        @tilefish.extern(ptx=str(ptx), symbol="axpy", block=(128,), grid=lambda n: (n,), num_ctas=2)
        def nope(x: Tensor[n, f32], y: InOut[Tensor[n, f32]], n_elem: n): ...


# -- a global scratch -----------------------------------------------------------------------------------------------------------------------------------


def test_a_global_scratch_is_declared_built_recorded_and_given_per_cta(world, tmp_path):
    _, compiler = world
    compiler.global_scratch, compiler.global_scratch_align = 256, 128  # two device-side TMA descriptors per program
    op = flat("desc", global_scratch=256)
    (spec,) = compile_specs(op)
    assert spec.global_scratch == 256 and spec.to_dict()["global_scratch"] == 256
    built = build(spec, arch=ARCH)
    assert (built.launch.global_scratch_size, built.launch.global_scratch_align) == (256, 128) and built.launch.replay_blockers() == []
    write_manifest([built], tmp_path / "out" / "manifest.json", cubins=True)
    entry = json.loads((tmp_path / "out" / "manifest.json").read_text())["kernels"][0]
    assert entry["launch"]["global_scratch_size"] == 256 and entry["launch"]["global_scratch_align"] == 128
    replayed = Manifest.load(tmp_path / "out").built_kernel(spec, f"sm_{ARCH}", pinned=True)
    assert stand_in(replayed).scratch_bytes((7,)) == 7 * 256
    # with clusters: a program is num_ctas CTAs, each its share
    compiler.cluster_dims = None
    clustered = build(compile_specs(flat("desc", global_scratch=256, num_ctas=2))[0], arch=ARCH)
    assert stand_in(clustered).scratch_bytes((7,)) == 7 * 2 * 256
    assert declaration_digest(op) != declaration_digest(flat("desc"))


def test_a_global_scratch_that_is_not_the_declared_one_is_refused(world):
    _, compiler = world
    compiler.global_scratch, compiler.global_scratch_align = 256, 128
    with pytest.raises(TilefishBuildError, match="addresses 256 bytes of global scratch per CTA, but declare global_scratch=256"):
        build(compile_specs(flat("desc"))[0], arch=ARCH)
    with pytest.raises(TilefishBuildError, match="its declaration says global_scratch=128: declare global_scratch=256"):
        build(compile_specs(flat("desc", global_scratch=128))[0], arch=ARCH)
    compiler.global_scratch_align = 512
    with pytest.raises(TilefishBuildError, match="aligned to 512 bytes"):
        build(compile_specs(flat("desc", global_scratch=256))[0], arch=ARCH)
    for bad in (-1, 1 << 21, True):
        with pytest.raises(TilefishDeclarationError, match="global_scratch"):
            flat("desc", global_scratch=bad)


def test_a_launch_is_given_the_global_scratch_it_needs_and_no_other():
    (spec,) = compile_specs(flat("desc", global_scratch=256))
    k = stand_in(fake_build(spec, arch=f"sm_{ARCH}"))
    args = [("pointer", 1 << 40), ("pointer", 1 << 41), ("scalar", 4096)]
    with pytest.raises(TilefishRuntimeError, match="give it a scratch"):
        k.launch((4,), args, 0)
    with pytest.raises(TilefishRuntimeError, match="256-byte aligned"):
        k.launch((4,), args, 0, scratch=(1 << 42) + 128)
    plain = stand_in(fake_build(compile_specs(flat())[0], arch=f"sm_{ARCH}"))
    with pytest.raises(TilefishRuntimeError, match="has no global scratch: give it none"):
        plain.launch((4,), args, 0, scratch=1 << 42)


def test_the_reader_holds_a_global_scratch_to_its_rules(world, tmp_path):
    _, compiler = world
    compiler.global_scratch, compiler.global_scratch_align = 256, 128
    write_manifest([build(compile_specs(flat("desc", global_scratch=256))[0], arch=ARCH)], tmp_path / "manifest.json", cubins=True)
    path = tmp_path / "manifest.json"
    good = json.loads(path.read_text())
    for change in (
        lambda e: e["launch"].update(global_scratch_align=96),
        lambda e: e["launch"].update(global_scratch_size=0),
        lambda e: e["launch"].update(global_scratch_align=1 << 32),  # (a power of two beyond the u32 the Rust reader reads)
    ):
        doc = copy.deepcopy(good)
        change(doc["kernels"][0])
        path.write_text(json.dumps(doc))
        with pytest.raises(TilefishManifestError, match="global_scratch_align"):
            Manifest.load(tmp_path)
    # a JSON null is as absent, as the Rust reader reads it
    doc = copy.deepcopy(good)
    doc["kernels"][0]["launch"].update(global_scratch_size=0, global_scratch_align=None)
    path.write_text(json.dumps(doc))
    Manifest.load(tmp_path)
    # a manifest of tilefish 0.0.6 records no alignment: it loads, and is refused when it is launched (export it again)
    doc = copy.deepcopy(good)
    del doc["kernels"][0]["launch"]["global_scratch_align"]
    assert LaunchAttributes.from_dict(doc["kernels"][0]["launch"]).replay_blockers() == [
        "global_scratch_size=256 of an alignment the manifest does not record (export it again)"
    ]


# -- a plan of a kernel with a global scratch -----------------------------------------------------------------------------------------------------------

M = Dim("M", buckets=[64, 1000])


@tilefish.extern(triton="lk:desc", num_warps=4, grid=lambda M, BLOCK: (tilefish.cdiv(M, BLOCK),), global_scratch=256, name="desc")
def desc(x: Tensor[M, 64, f32], y: Out[Tensor[M, 64, f32]], n_rows: M, BLOCK: const = 64): ...


@tilefish.extern(triton="lk:plain", num_warps=4, grid=lambda M: (M,), name="plain")
def plain(x: Tensor[M, 64, f32], y: Out[Tensor[M, 64, f32]], n_rows: M): ...


@tilefish.graph
def chain(x: Tensor[M, 64, f32]):
    return desc(plain(x))


class ScratchBackend(FakeBackend):
    """The fake backend, with loaded kernels that say what scratch a launch needs (as the real ones do), and launches that record it."""

    def load(self, built, index):
        self.log.append(("load", built.spec.stem))
        spec = built.spec
        return SimpleNamespace(stem=spec.stem, scratch_bytes=lambda grid: spec.global_scratch * math.prod(grid) * spec.num_ctas)

    def launch(self, loaded, grid, args, stream, scratch=None):
        self.log.append(("launch", loaded.stem, tuple(grid), tuple(args), stream, scratch))


def test_a_plan_gives_a_kernel_its_global_scratch_from_the_workspace_of_the_arena(tmp_path):
    plan = tilefish.plan(chain)
    ws = next(b for b in plan.doc["buffers"] if b["role"] == "workspace")
    assert ws["shape"] == [256 * 16] and ws["bytes"] == 256 * 16  # the largest bucket: cdiv(1000, 64) = 16 programs of one CTA, 256 bytes each
    for bucket in plan.doc["buckets"]:
        plain_step, desc_step = bucket["steps"]
        assert "workspace" not in plain_step and desc_step["workspace"] == ws["name"]
    _plan.validate_doc(plan.doc, made=True)
    import plan_fakes

    exported = plan_fakes.export(plan, tmp_path)  # check_against_manifest: the step has the workspace, its entry a global scratch
    be = ScratchBackend()
    runner = PlanRunner(exported, FakeProvider(), be)
    runner.run((FakeTensor((1000, 64), "f32", 1 << 36),), {})
    launches = [e for e in be.log if e[0] == "launch"]
    assert launches[0][5] is None and launches[1][5] == runner.arena_of(0).ptr + ws["offset"]
    # a document whose step has no workspace for a kernel with a scratch (or one for a kernel without) is not its manifest's
    doc = copy.deepcopy(exported.doc)
    for bucket in doc["buckets"]:
        bucket["steps"][1].pop("workspace")
    with pytest.raises(TilefishManifestError, match="has 256 bytes of global scratch per CTA, and the step has no workspace|workspace exactly when"):
        _plan.validate_doc(doc)
        _plan.check_against_manifest(doc, exported.manifest)


# -- the Rust emitter -----------------------------------------------------------------------------------------------------------------------------------


def test_the_rust_launchers_say_each_launch_shape_and_take_the_global_scratch_last(tmp_path):
    from test_rust_host_schema3 import build_manifest

    from tilefish.hosts import rust

    scratchy = desc
    pers = tiles(tilefish.persistent(programs_per_sm=2))
    manifest = build_manifest(tmp_path / "m", [scratchy, pers, flat(num_ctas=2)])
    source = rust.render(manifest, ops=[scratchy, pers, flat(num_ctas=2)])
    assert "global_scratch: TensorMut<'a, U8>," in source  # the last tensor of `launch` and `launch_raw`
    assert "unsafe { self.kernel.launch_raw(stream, &[x.into_arg(), y.into_arg(), global_scratch.into_arg()]) }" in source
    assert "A global scratch of 256 bytes per CTA" in source
    assert "A persistent grid: the grid is the tiles, and a launch runs at most 2 program(s) per SM" in source
    assert "Thread-block clusters of (2, 1, 1)" in source
