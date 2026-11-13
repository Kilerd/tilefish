"""An ``Out`` that may be written over an input: ``Out[Tensor[...], MayAlias("r")]`` (an in-place kernel with two pointers).

A hand-written kernel ``k(x, r, y, r_out)`` that reads every element of ``r`` before it writes the same element of ``r_out`` may be launched with one tensor
for ``r`` and ``r_out``. The rule:

* **the declaration** (an extern only) names a tensor input of the op with the dtype and the declared shape of the ``Out``, not strided and not a device
  scalar, and one ``Out`` at most names an input; it reaches the compile spec, the manifest's ABI (``"alias": "r"`` on the ``out`` pointer) and the
  declaration digest (a declaration without it digests as it always did);
* **a launch** (Python, a plan step, Rust) accepts ``r_out`` and ``r`` that are exactly one tensor and refuses any other overlap;
* **a plan** writes the result over ``r`` when nothing reads that version of ``r`` after the call: ``r`` is a whole intermediate (not a parameter,
  not a view), no later node reads its storage (a view, a version, the return) and no other argument of the call is in it. Otherwise the result has
  a buffer of its own: the values are the same, only the memory differs.

Nothing here needs a GPU (``test_gpu_alias.py`` runs the kernels).
"""

import json

import plan_fakes
import pytest
from fake_torch import FakeTensor

import tilefish
from tilefish import (
    DevScalar,
    Dim,
    InOut,
    MayAlias,
    Out,
    Strided,
    Tensor,
    TilefishDeclarationError,
    TilefishManifestError,
    TilefishPreconditionError,
    TilefishSpecError,
    bf16,
    const,
    f32,
)
from tilefish import _plan, aot
from tilefish._launch import TensorInfo, check_no_alias
from tilefish._manifest import Manifest

Rb = Dim("Rb", buckets=[8, 24])
N = Dim("N")


@tilefish.extern(triton="gpu_kernels_p5:add_into_kernel", num_warps=1, grid=lambda Rb: (Rb,))
def add_into(
    x: Tensor[Rb, 64, f32], r: Tensor[Rb, 64, f32], y: Out[Tensor[Rb, 64, f32]], r_out: Out[Tensor[Rb, 64, f32], MayAlias("r")], D: const = 64
): ...


@tilefish.extern(triton="gpu_kernels_p5:add_into_kernel", num_warps=1, grid=lambda Rb: (Rb,), name="add_apart")
def add_apart(x: Tensor[Rb, 64, f32], r: Tensor[Rb, 64, f32], y: Out[Tensor[Rb, 64, f32]], r_out: Out[Tensor[Rb, 64, f32]], D: const = 64): ...


@tilefish.kernel
def twice(x: Tensor[Rb, 64, f32]) -> Tensor[Rb, 64, f32]:
    return x * 2.0


@tilefish.kernel
def plus(x: Tensor[Rb, 64, f32], y: Tensor[Rb, 64, f32]) -> Tensor[Rb, 64, f32]:
    return x + y


@tilefish.graph
def chain(x: Tensor[Rb, 64, f32]):
    r = twice(x)  # an intermediate
    y1, r = add_into(x, r)  # in place: nothing reads the r of before after this call
    y2, r = add_into(y1, r)  # in place again: the result of the first call is written over
    return y2, r


def binding(plan, step, name):
    return next(a["binding"] for a in plan.steps(Rb=8)[step]["args"] if a["name"] == name)


# -- the declaration ---------------------------------------------------------------------------------------------------------------------------


def test_the_out_names_the_input_and_the_spec_and_the_digest_carry_it():
    p = add_into.signature.params[3]
    assert p.alias == "r" and str(p) == "r_out: Out[Tensor[Rb, 64, f32], MayAlias('r')]"
    assert repr(Out[Tensor[Rb, f32], MayAlias("r")]) == "Out[Tensor[Rb, f32], MayAlias('r')]"
    (spec,) = aot.compile_specs(add_into)
    by = {a.name: a for a in spec.args}
    assert by["r_out"].alias == "r" and by["y"].alias is None and by["r"].alias is None
    assert spec.to_dict()["args"][3]["alias"] == "r" and "alias" not in spec.to_dict()["args"][2]
    assert aot.CompileSpec.from_dict(spec.to_dict()) == spec
    (apart,) = aot.compile_specs(add_apart)
    assert "alias" not in json.dumps(apart.to_dict())
    # the declaration digest has it: the same kernel without it is another declaration (and one without it digests as it always did)
    assert spec.declaration_digest != aot.compile_specs(add_apart)[0].declaration_digest


@pytest.mark.parametrize(
    "make, message",
    [
        (lambda: Out[Tensor[N, f32], MayAlias("1x")], "MayAlias names the tensor parameter"),
        (lambda: InOut[Tensor[N, f32], MayAlias("x")], "only an Out tensor names an input"),
    ],
)
def test_a_malformed_annotation_is_refused(make, message):
    with pytest.raises(TilefishDeclarationError, match=message):
        make()


def declare(fn, **kw):
    return tilefish.extern(triton="m:k", grid=lambda N: (N,), **kw)(fn)


def test_what_an_out_may_alias_is_checked():
    def missing(x: Tensor[N, f32], y: Out[Tensor[N, f32], MayAlias("z")]): ...

    def an_out(y0: Out[Tensor[N, f32]], y: Out[Tensor[N, f32], MayAlias("y0")]): ...

    def another_dtype(x: Tensor[N, bf16], y: Out[Tensor[N, f32], MayAlias("x")]): ...

    def another_shape(x: Tensor[N, 2, f32], y: Out[Tensor[2, N, f32], MayAlias("x")]): ...

    def strided(x: Strided[Tensor[N, f32]], y: Out[Tensor[N, f32], MayAlias("x")]): ...

    def scalar(x: DevScalar[f32], y: Out[Tensor[1, f32], MayAlias("x")]): ...

    def twice_over(x: Tensor[N, f32], y: Out[Tensor[N, f32], MayAlias("x")], z: Out[Tensor[N, f32], MayAlias("x")]): ...

    for fn, message in [
        (missing, "may alias 'z', which is not a tensor input"),
        (an_out, "may alias 'y0', which is not a tensor input"),
        (another_dtype, "which has another dtype or shape"),
        (another_shape, "which has another dtype or shape"),
        (strided, "which is a strided view"),
        (scalar, "which is a DevScalar|another dtype or shape"),
        (twice_over, "both may alias 'x'"),
    ]:
        with pytest.raises(TilefishDeclarationError, match=message):
            declare(fn)


def test_only_an_extern_declares_it():
    with pytest.raises(TilefishDeclarationError, match="only an extern op can declare"):

        @tilefish.kernel
        def k(x: Tensor[N, f32], y: Out[Tensor[N, f32], MayAlias("x")]):
            y[...] = x


# -- the spec and the manifest readers ---------------------------------------------------------------------------------------------------------


def spec_doc():
    (spec,) = aot.compile_specs(add_into)
    return spec.to_dict()


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d["args"][3].__setitem__("alias", "nope"), "is not an `in` pointer"),
        (lambda d: d["args"][3].__setitem__("alias", "y"), "is not an `in` pointer"),
        (lambda d: d["args"][3].__setitem__("alias", 3), "alias must be the name"),
        (lambda d: d["args"][2].__setitem__("alias", "r"), "both may alias 'r'"),
        (lambda d: d["args"][0].__setitem__("alias", "r"), "only an out pointer may alias"),
        (lambda d: d["args"][1]["shape_dims"].__setitem__(1, 32), "another dtype or declared shape|shape_dims"),
        (lambda d: d["args"][4].__setitem__("alias", "r"), "must have exactly the fields"),
    ],
)
def test_the_spec_reader_refuses_an_alias_that_does_not_fit(change, message):
    d = spec_doc()
    change(d)
    with pytest.raises(TilefishSpecError, match=message):
        aot.CompileSpec.from_dict(d)


def manifest_dir(tmp_path):
    from built_fakes import fake_build
    from tilefish._build import write_manifest_doc

    (spec,) = aot.compile_specs(add_into)
    write_manifest_doc([fake_build(spec, arch="sm_103")], tmp_path / "manifest.json", cubins=True)
    return tmp_path


def test_the_manifest_has_the_alias_on_the_out_pointer_and_its_reader_holds_it_to_the_rules(tmp_path):
    d = manifest_dir(tmp_path)
    m = Manifest.load(d)
    abi = m.get("add_into", "sm_103").entry["abi"]["params"]
    assert abi[3]["alias"] == "r" and all("alias" not in p for p in abi[:3])
    doc = json.loads((d / "manifest.json").read_text())

    def refused(change, message):
        bad = json.loads(json.dumps(doc))
        change(bad["kernels"][0]["abi"]["params"])
        (d / "manifest.json").write_text(json.dumps(bad))
        with pytest.raises(TilefishManifestError, match=message):
            Manifest.load(d)

    refused(lambda ps: ps[3].__setitem__("alias", "x0"), "is not an `in` pointer")
    refused(lambda ps: ps[3].__setitem__("alias", 7), "alias must be a str")
    refused(lambda ps: ps[2].__setitem__("alias", "r"), "both may alias 'r'")
    refused(lambda ps: ps[0].__setitem__("alias", "r"), "only an out pointer may alias")
    refused(lambda ps: ps[1].__setitem__("shape", [32, 64]), "another dtype or declared shape")


# -- the launch check ----------------------------------------------------------------------------------------------------------------------------


def info(ptr, shape=(8, 64), strides=None):
    contiguous = strides is None
    strides = strides or (shape[1], 1)
    return TensorInfo(shape=shape, dtype="torch.float32", ptr=ptr, contiguous=contiguous, device="cuda:0", nbytes=8 * 64 * 4, strides=strides, itemsize=4)


def test_a_launch_takes_the_out_and_its_input_as_exactly_one_tensor_and_nothing_else():
    infos = {"x": info(1 << 20), "r": info(2 << 20), "y": info(3 << 20), "r_out": info(2 << 20)}
    check_no_alias(infos, {"y", "r_out"}, {"r_out": "r"})  # exactly r: accepted
    with pytest.raises(TilefishPreconditionError, match="argument 'r_out' is written in place but overlaps argument 'r'"):
        check_no_alias(infos, {"y", "r_out"})  # not declared
    shifted = {**infos, "r_out": info((2 << 20) + 256)}
    with pytest.raises(TilefishPreconditionError, match="only as exactly that tensor") as e:
        check_no_alias(shifted, {"y", "r_out"}, {"r_out": "r"})
    assert e.value.kind == "alias" and e.value.arg == "r_out"
    over_x = {**infos, "r_out": info(1 << 20)}
    with pytest.raises(TilefishPreconditionError, match="'r_out' is written in place but overlaps argument 'x'"):
        check_no_alias(over_x, {"y", "r_out"}, {"r_out": "r"})
    y_over_r = {**infos, "y": info(2 << 20)}
    with pytest.raises(TilefishPreconditionError, match="'y' is written in place but overlaps argument 'r'"):
        check_no_alias(y_over_r, {"y", "r_out"}, {"r_out": "r"})


def test_a_single_launch_may_be_given_the_input_as_the_out():
    from fake_torch import prepare

    x, r, y = FakeTensor((8, 64), "f32", 1 << 36), FakeTensor((8, 64), "f32", 2 << 36), FakeTensor((8, 64), "f32", 3 << 36)
    prep, error = prepare(add_into, (x, r), next_slab=8 << 36, outputs=(y, r))
    assert error is None and prep.tensors["r_out"] is r
    _, error = prepare(add_apart, (x, r), next_slab=8 << 36, outputs=(y, r))
    assert isinstance(error, TilefishPreconditionError) and "overlaps argument 'r'" in str(error)
    _, error = prepare(add_into, (x, r), next_slab=8 << 36, outputs=(r, r))  # y over r: no declaration says it may
    assert isinstance(error, TilefishPreconditionError) and error.kind == "alias"


# -- a plan ----------------------------------------------------------------------------------------------------------------------------------------


def test_a_plan_writes_the_result_over_an_intermediate_nothing_reads_after():
    p = tilefish.plan(chain)
    names = [b["name"] for b in p.buffers]
    steps = p.steps(Rb=8)
    assert [s["entry"] for s in steps] == ["twice", "add_into", "add_into"]
    made = binding(p, 0, "out")
    # both calls write r_out over r: one buffer for the three versions of the residual, the one the graph returns
    assert binding(p, 1, "r") == binding(p, 1, "r_out") == made
    assert binding(p, 2, "r") == binding(p, 2, "r_out") == made
    assert p.outputs == [binding(p, 2, "y"), made] and names.count(made) == 1
    roles = {b["name"]: b["role"] for b in p.buffers}
    assert roles[made] == "output" and sum(1 for r in roles.values() if r == "intermediate") == 1  # y1 only
    # the written-before-read rule and the arena's rules hold (tilefish.plan checks every plan it makes)
    _plan.validate_doc(p.doc, made=True)


def test_a_result_gets_a_buffer_of_its_own_when_the_input_is_read_after_the_call():
    @tilefish.graph
    def read_after(x: Tensor[Rb, 64, f32]):
        r = twice(x)
        y, r2 = add_into(x, r)
        return plus(y, r), r2  # the r of before is read after the call: r2 cannot be written over it

    p = tilefish.plan(read_after)
    assert binding(p, 1, "r") != binding(p, 1, "r_out")


def test_a_parameter_a_view_a_returned_value_and_an_argument_twice_are_not_written_over():
    @tilefish.graph
    def over_parameter(x: Tensor[Rb, 64, f32], r: Tensor[Rb, 64, f32]):
        return add_into(x, r)  # r is the caller's

    @tilefish.graph
    def over_view(x: Tensor[Rb, 64, f32]):
        r = twice(x).reshape(Rb, 2, 32).reshape(Rb, 64)  # a view of the result
        return add_into(x, r)

    @tilefish.graph
    def over_returned(x: Tensor[Rb, 64, f32]):
        r = twice(x)
        y, r2 = add_into(x, r)
        return y, r2, r

    @tilefish.graph
    def twice_in_call(x: Tensor[Rb, 64, f32]):
        r = twice(x)
        return add_into(r, r)  # x and r are one buffer: writing r_out over it would overwrite what x reads

    for g in (over_parameter, over_view, over_returned, twice_in_call):
        p = tilefish.plan(g)
        steps = p.steps(Rb=8)
        call = next(i for i, s in enumerate(steps) if s["entry"] == "add_into")
        assert binding(p, call, "r_out") != binding(p, call, "r"), g.name


def test_an_op_without_the_declaration_is_never_written_over():
    @tilefish.graph
    def apart(x: Tensor[Rb, 64, f32]):
        r = twice(x)
        return add_apart(x, r)

    p = tilefish.plan(apart)
    assert binding(p, 1, "r_out") != binding(p, 1, "r")


def test_the_in_place_plan_runs_and_its_reuse_keeps_the_rules(tmp_path):
    for reuse in (False, True):
        p = tilefish.plan(chain, reuse=reuse)
        exported = plan_fakes.export(p, tmp_path / f"r{reuse}")
        assert exported.digest == p.with_manifest("sm_103", exported.manifest_digest).digest
        doc = exported.doc
        out = [b for b in doc["buffers"] if b["role"] == "output"]
        assert len(out) == 2


def test_the_runner_launches_the_in_place_step_with_one_address_for_both():
    from test_planrun import FakeBackend, FakeProvider

    from tilefish._planrun import PlanRunner

    p = tilefish.plan(chain)
    be = FakeBackend()
    r = PlanRunner(p, FakeProvider(), be)
    x = FakeTensor((8, 64), "f32", 1 << 36)
    y, res = r.run((x,), {})
    launches = be.launches()
    pointers = [[v for k, v in e[3] if k == "pointer"] for e in launches]
    assert pointers[1][1] == pointers[1][3] == res._ptr  # r and r_out of the first add_into: the returned residual's memory
    assert pointers[2][1] == pointers[2][3] == res._ptr and pointers[2][2] == y._ptr
