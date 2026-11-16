"""The memory and the time of an export of many buckets, with no Triton and no GPU (``tilefish._ir.frontend``, ``tilefish._build.export_session``).

An export runs Triton's frontend for the IR digest of every binary it builds. The buckets of one ``by_bucket=`` configuration are one kernel with
one IR, so an export runs the frontend once per distinct input (the function, the signature, the attributes, the constants, the options and the
architecture), not once per bucket; the entries that have the same binary share one copy of its cubin and its PTX; and a traced export keeps the
stages of its builds in files until it writes the trace. None of it changes a byte of what is written: every test here compares the export with
the one each bucket's own frontend makes (no session). ``fake_compile`` stands in for the frontends and the compiler (``test_build_cache.py`` says
how); ``test_gpu_frontend_memory.py`` measures the memory with Triton on a GPU, and ``irmem_export.py`` the export of a large kernel.
"""

import contextlib
import hashlib
import json
import pathlib
import sys

import fake_compile
import pytest

import tilefish
from tilefish import Dim, Out, Tensor, TilefishBuildError, _build, _ir, _sass, _trace, _tf32guard, const, f32, runtime
from tilefish.aot import compile_specs

ARCH = 103
T = Dim("T", buckets=[8, 16, 24, 32])
D = Dim("D", buckets=[128, 256, 384, 512, 640, 768])

KERNELS = '''from fake_jit import gluon_jit

from tilefish import _probe


@gluon_jit
def chain(x, y, n_t, n_d, BLOCK, STEPS, WARPS):
    return (x * n_d + y, BLOCK, STEPS, WARPS)


@gluon_jit
def probed(x, y, n_t, n_d, BLOCK, STEPS, WARPS):
    # tilefish.probe("scaled") as Triton's code generator meets it: the frontend of an instrumented build emits it into the IR (it has a table of
    # probe ids then, tilefish._probe.frontend), a production one emits nothing
    mark = "probe scaled" if getattr(_probe._state, "registry", None) is not None else ""
    return (x * n_d + y, BLOCK, STEPS, WARPS, mark)
'''


def config(T, D):
    """Three configurations over the 24 buckets: 8, 8 and 8 of them."""
    if D <= 256:
        return tilefish.config(num_warps=2, BLOCK=128, WARPS=2)
    if D <= 512:
        return tilefish.config(num_warps=4, BLOCK=256, WARPS=4)
    return tilefish.config(num_warps=8, BLOCK=512, WARPS=8)


def declare(name="chain"):
    @tilefish.extern(gluon="fo_kernels:chain", num_warps=4, grid=lambda T, D, BLOCK: (T, tilefish.cdiv(D, BLOCK)), by_bucket=config, name=name)
    def chain(x: Tensor[T, D, f32], y: Out[Tensor[T, D, f32]], n_t: T, n_d: D, BLOCK: const = 256, STEPS: const = 64, WARPS: const = 4): ...

    return chain


def declare_probed(name="probed"):
    @tilefish.extern(gluon="fo_kernels:probed", num_warps=4, grid=lambda T, D, BLOCK: (T, tilefish.cdiv(D, BLOCK)), by_bucket=config, name=name)
    def probed(x: Tensor[T, D, f32], y: Out[Tensor[T, D, f32]], n_t: T, n_d: D, BLOCK: const = 256, STEPS: const = 64, WARPS: const = 4): ...

    return probed


@pytest.fixture
def world(tmp_path, monkeypatch):
    compiler = fake_compile.install(monkeypatch)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(runtime, "_modules", {})
    monkeypatch.delenv(_trace.TRACE_ENV, raising=False)
    directory = tmp_path / "project"
    directory.mkdir()
    (directory / "fo_kernels.py").write_text(KERNELS)
    monkeypatch.syspath_prepend(str(directory))
    sys.modules.pop("fo_kernels", None)
    yield compiler
    sys.modules.pop("fo_kernels", None)


def files(directory):
    """Every file under ``directory`` (but the trace's), by its path relative to it: its bytes."""
    root = pathlib.Path(directory)
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file() and "trace" not in p.relative_to(root).parts}


@contextlib.contextmanager
def per_bucket(monkeypatch):
    """An export as it was before its session: the frontend of every bucket, a copy of every binary, the stages in memory."""
    with monkeypatch.context() as m:
        m.setattr(runtime, "export_session", contextlib.nullcontext)
        yield


def use_listing(monkeypatch):
    text = '\t.target\tsm_103a\n\t.section\t.text.tf_k,"ax",@progbits\n        /*0000*/                   EXIT ;\n'

    def disassemble(cubin, *, ptx=None, options=None, prebuilt=False):
        return _sass.Disassembly(text, "", _sass.parse_listing(text), "sm_103a", False, "/x/nvdisasm", "13.5.37", "sm_103a")

    monkeypatch.setattr(_sass, "disassemble", disassemble)


# -- an export ---------------------------------------------------------------------------------------------------------------------------------


def test_an_export_runs_the_frontend_once_per_distinct_configuration_and_writes_the_same_bytes(world, tmp_path, monkeypatch):
    compiler = world
    op = declare()
    specs = compile_specs(op)
    assert len(specs) == 24 and len({json.dumps(sorted(s.ast_source_inputs().options.items())) for s in specs}) == 3
    exported = runtime.export(op, tmp_path / "once", arch=ARCH)
    # three digests (one per configuration) and three compiles (each runs the frontend once more): the other 21 buckets are cache hits
    assert (compiler.frontends, compiler.count) == (3 + 3, 3)
    doc = json.loads((exported.directory / "manifest.json").read_text())
    assert len(doc["kernels"]) == 24 and len({e["source_digest"] for e in doc["kernels"]}) == 3

    compiler.frontends = 0
    with per_bucket(monkeypatch):
        before = runtime.export(op, tmp_path / "per-bucket", arch=ARCH)
    assert compiler.frontends == 24 and compiler.count == 3  # each bucket's own frontend: what an export did before (the cache is warm now)
    assert before.digest == exported.digest
    assert files(tmp_path / "once") == files(tmp_path / "per-bucket")  # the manifest and every cubin, byte for byte

    compiler.frontends = 0
    runtime.export(op, tmp_path / "again", arch=ARCH)  # a session ends with its export: the next one runs the frontend again
    assert compiler.frontends == 3 and files(tmp_path / "again") == files(tmp_path / "once")


def test_the_entries_of_one_binary_share_one_copy_of_it_while_the_export_runs(world, tmp_path, monkeypatch):
    seen = []
    original = _build.write_manifest_doc

    def spy(built, *a, **k):
        seen.extend(built)
        return original(built, *a, **k)

    monkeypatch.setattr(runtime, "write_manifest_doc", spy)
    runtime.export(declare(), tmp_path / "out", arch=ARCH)
    assert len(seen) == 24
    assert len({b.sha256 for b in seen}) == len({id(b.cubin) for b in seen}) == 3  # one object per binary, not one per entry
    assert len({id(b.ptx) for b in seen}) == len({b.ptx for b in seen})  # (and per PTX: the fake compiler's are one text)


def test_a_traced_export_keeps_its_stages_in_files_and_writes_the_same_trace(world, tmp_path, monkeypatch):
    compiler = world
    use_listing(monkeypatch)
    items = []
    original = _trace.write

    def spy(directory, traced, *a, **k):
        if not items:  # the first export: in a session, every stage is a file of it, read back when it is written
            items.extend(traced)
            for item in traced:
                assert isinstance(item.built.stages, _build.SpilledStages)
                assert all(p.is_file() for p in item.built.stages._files.values())
        return original(directory, traced, *a, **k)

    monkeypatch.setattr(_trace, "write", spy)
    op = declare()
    runtime.export(op, tmp_path / "traced", arch=ARCH, trace=True)
    assert compiler.count == 24  # a traced export compiles every binary (the cache holds no IR), as before
    assert compiler.frontends == 3 + 24  # but the digests are one per configuration
    spilled = {p for item in items for p in item.built.stages._files.values()}
    assert spilled and not any(p.exists() for p in spilled)  # the directory goes with the session
    assert len(items) == 24 and len(spilled) < sum(len(item.built.stages._files) for item in items)  # one file per distinct text

    with per_bucket(monkeypatch):
        runtime.export(op, tmp_path / "plain", arch=ARCH, trace=True)
    assert files(tmp_path / "traced") == files(tmp_path / "plain")
    a, b = tmp_path / "traced" / "trace", tmp_path / "plain" / "trace"
    assert {str(p.relative_to(a)): p.read_bytes() for p in a.rglob("*") if p.is_file()} == {
        str(p.relative_to(b)): p.read_bytes() for p in b.rglob("*") if p.is_file()
    }


def test_a_build_outside_an_export_keeps_its_stages_in_memory(world):
    spec = compile_specs(declare())[0]
    built = _build.build(spec, arch=ARCH, stages=True)
    assert type(built.stages) is dict and built.stages["ptx"] == built.ptx


def test_spilled_stages_after_their_session_say_so(world, tmp_path):
    spec = compile_specs(declare())[0]
    with _build.export_session():
        built = _build.build(spec, arch=ARCH, stages=True)
        assert dict(built.stages)["ptx"] == built.ptx and len(built.stages) == len(list(built.stages))
    with pytest.raises(TilefishBuildError, match="for the export it was built in, which is over"):
        built.stages["ptx"]


# -- the frontend, once per input --------------------------------------------------------------------------------------------------------------


def test_a_scope_runs_the_frontend_once_per_input_and_never_outside(world):
    compiler = world
    spec = compile_specs(declare())[0]
    jit = runtime.resolve_triton(spec.triton)
    inputs = spec.ast_source_inputs()
    first = _ir.frontend(jit, inputs, arch=ARCH, options=inputs.options)
    assert first.digest == _ir.ir_digest(_ir.ttir_text(jit, inputs, arch=ARCH, options=inputs.options), "gluon")
    compiler.frontends = 0
    _ir.frontend(jit, inputs, arch=ARCH, options=inputs.options)
    _ir.frontend(jit, inputs, arch=ARCH, options=inputs.options)
    assert compiler.frontends == 2  # outside a scope: every call (a jit call, a replay call)
    compiler.frontends = 0
    with _ir.frontend_scope():
        again = _ir.frontend(jit, spec.ast_source_inputs(), arch=ARCH, options=inputs.options)
        with _ir.frontend_scope():  # a scope in a scope is the outer one
            _ir.frontend(jit, spec.ast_source_inputs(), arch=ARCH, options=dict(inputs.options))
        assert compiler.frontends == 1 and again == first
        _ir.frontend(jit, inputs, arch=90, options=inputs.options)  # another architecture
        _ir.frontend(jit, inputs, arch=ARCH, options={**inputs.options, "num_warps": 8})  # other options
        assert compiler.frontends == 3
    _ir.frontend(jit, inputs, arch=ARCH, options=inputs.options)
    assert compiler.frontends == 4  # the scope is over


@pytest.mark.parametrize(
    "a, b",
    [(1, 1.0), (1, True), (0.0, -0.0), ([1, 2], (1, 2)), ("1", 1), (None, 0)],
)
def test_constants_that_print_alike_are_different_inputs(world, a, b):
    compiler = world
    spec = compile_specs(declare())[0]
    jit = runtime.resolve_triton(spec.triton)
    inputs = spec.ast_source_inputs()
    key = next(iter(inputs.constants))
    with _ir.frontend_scope():
        for value in (a, b, a):
            inputs.constants[key] = value
            _ir.frontend(jit, inputs, arch=ARCH, options=inputs.options)
    assert compiler.frontends == 2


def test_an_input_that_is_not_plain_is_not_shared_and_a_failure_is_not_kept(world):
    compiler = world
    spec = compile_specs(declare())[0]
    jit = runtime.resolve_triton(spec.triton)
    inputs = spec.ast_source_inputs()

    class Opaque:  # a value whose repr may hide what it holds
        def __repr__(self):
            return "x"

    inputs.constants[next(iter(inputs.constants))] = Opaque()
    with _ir.frontend_scope():
        _ir.frontend(jit, inputs, arch=ARCH, options=inputs.options)
        _ir.frontend(jit, inputs, arch=ARCH, options=inputs.options)
        assert compiler.frontends == 2
        good = spec.ast_source_inputs()
        compiler.fail_frontend = RuntimeError("no")
        with pytest.raises(TilefishBuildError):
            _ir.frontend(jit, good, arch=ARCH, options=good.options)
        compiler.fail_frontend = None
        _ir.frontend(jit, good, arch=ARCH, options=good.options)
        assert compiler.frontends == 4


def test_a_production_and_an_instrumented_build_of_one_kernel_in_one_session_have_their_own_ir(world):
    """The IR of an instrumented build has its probes, so the variant is part of the input a session runs the frontend once for: in one export
    session the two builds of a kernel get the IR and the digest each gets alone, in either order."""
    compiler = world
    spec = compile_specs(declare_probed())[0]
    cfg = tilefish.instrument(capacity=8)
    plain = _build.build(spec, arch=ARCH)
    inst = _build.build(spec, arch=ARCH, instrument=cfg)
    assert inst.ir_digest != plain.ir_digest and inst.sha256 != plain.sha256 and inst.instrument is not None and plain.instrument is None
    for order in ((None, cfg, None, cfg), (cfg, None, cfg, None)):
        frontends, compiles = compiler.frontends, compiler.count
        with _build.export_session():
            got = [_build.build(spec, arch=ARCH, instrument=c) for c in order]
        for c, b in zip(order, got):
            want = plain if c is None else inst
            assert (b.ir_digest, b.sha256, b.instrument) == (want.ir_digest, want.sha256, want.instrument)
        # one digest per variant; the instrumented builds compile every time (never from the build cache), each running the frontend once more
        assert compiler.count - compiles == 2 and compiler.frontends - frontends == 2 + 2


def test_an_export_that_instruments_one_of_two_ops_of_one_kernel_writes_what_each_bucket_alone_writes(world, tmp_path, monkeypatch):
    """Two operators of one kernel function, one of them instrumented (``tilefish.instrument(ops=...)``): in the export's session each keeps its own
    IR, and the export is the one the frontend of every bucket makes alone."""
    a, b = declare_probed("probed_a"), declare_probed("probed_b")
    cfg = tilefish.instrument(capacity=8, ops=["probed_a"])
    exported = runtime.export([a, b], tmp_path / "once", arch=ARCH, instrument=cfg)
    kernels = exported.doc["kernels"]
    digests = {op: {e["source_digest"] for e in kernels if e["op"] == op} for op in ("probed_a", "probed_b")}
    assert all("instrument" in e for e in kernels if e["op"] == "probed_a") and not any("instrument" in e for e in kernels if e["op"] == "probed_b")
    assert len(digests["probed_a"]) == len(digests["probed_b"]) == 3 and not digests["probed_a"] & digests["probed_b"]
    with per_bucket(monkeypatch):
        alone = runtime.export([a, b], tmp_path / "per-bucket", arch=ARCH, instrument=cfg)
    assert alone.digest == exported.digest and files(tmp_path / "once") == files(tmp_path / "per-bucket")


def test_the_prepare_of_a_plan_loads_every_bucket_in_one_scope():
    """``prepare`` builds (``jit``) or loads and checks (``replay``) the binaries of every bucket of a plan in one scope: a replay of many buckets
    digests each kernel once. A run outside ``prepare`` is not in one."""
    import plan_model as pm
    from test_planrun import FakeBackend, FakeProvider

    from tilefish._planrun import PlanRunner

    scoped = []

    class Provider(FakeProvider):
        def kernel(self, callee, spec, index):
            scoped.append(getattr(_ir._scope, "table", None) is not None)
            return super().kernel(callee, spec, index)

    r = PlanRunner(tilefish.plan(pm.block), Provider(), FakeBackend())
    r.prepare(0)
    assert scoped and all(scoped)
    assert getattr(_ir._scope, "table", None) is None


# -- freeing a module ----------------------------------------------------------------------------------------------------------------------------


class Block:
    def __init__(self):
        self.erased = 0

    def erase(self):
        self.erased += 1


class Operation:
    def __init__(self, name, block=None):
        self.name, self.block = name, block

    def get_name(self):
        return self.name

    def get_block(self):
        return self.block


class Module:
    """What Triton's binding gives of a module: a post-order walk (every nested operation, then the top-level ones, then the module)."""

    def __init__(self, ops):
        self.ops = ops

    def walk(self, visit):
        for op in self.ops:
            visit(op)


def test_release_erases_the_body_of_a_module():
    body = Block()
    inner = Operation("arith.addf", Block())
    module = Module([inner, Operation("tt.func", body), Operation("builtin.module")])
    assert _ir.release_module(module) is True and body.erased == 1 and inner.block.erased == 0


def test_release_leaves_a_module_it_cannot_read():
    assert _ir.release_module(object()) is False  # no walk: another Triton, or a fake
    body = Block()
    assert _ir.release_module(Module([Operation("tt.func", body), Operation("tt.return")])) is False and body.erased == 0  # not a module last
    assert _ir.release_module(Module([Operation("builtin.module")])) is False  # an empty module: nothing to free
    assert _ir.release_module(Module([Operation("tt.func", None), Operation("builtin.module")])) is False

    class Broken(Module):
        def walk(self, visit):
            raise TypeError("incompatible function arguments")

    assert _ir.release_module(Broken([])) is False


# -- what is kept of the IR ------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "module {\n}\n", "é" * (3 << 20) + " x" * 1000 + "tt.dot", "a\nb\r\nc" * 400000])
def test_the_digest_is_hashed_in_pieces_and_is_the_digest_of_the_whole(text):
    for language, tag in (("triton", b"tilefish.triton-ir-digest.v1\0"), ("gluon", b"tilefish.gluon-ir-digest.v1\0")):
        assert _ir.ir_digest(text, language) == hashlib.sha256(tag + text.encode("utf-8")).hexdigest()


def test_the_tf32_guard_judges_the_kept_dots_as_it_judges_the_text():
    import test_tf32guard as t

    for frontend in (t.FRONTEND_IEEE, t.FRONTEND_IEEE_M32, t.FRONTEND_TF32, t.FRONTEND_TF32X3, t.FRONTEND_MIXED, "module {\n}\n"):
        for ptx, kw in (
            (t.PTX_TCGEN05_TF32, {"ttir": t.TTIR_IEEE, "ttgir": t.TTGIR_IEEE}),
            (t.PTX_TCGEN05_TF32, {}),
            (t.PTX_FMA, {"ttgir": t.TTGIR_FMA}),
            (t.PTX_MIXED, {"ttir": t.TTIR_MIXED, "ttgir": t.TTGIR_MIXED_TRAPPED}),
        ):
            ops = _tf32guard.frontend_ops(frontend)
            assert _tf32guard.problem(ops, ptx, **kw) == _tf32guard.problem(frontend, ptx, **kw)
    assert _tf32guard.frontend_ops("module {\n  tt.func @k() {\n  }\n}\n") == []
