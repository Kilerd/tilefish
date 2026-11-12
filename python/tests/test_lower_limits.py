"""i32 offsets: static overflow is rejected, dynamic sizes get a ``max_elements`` precondition; and how a host
computes the element count ``n`` of an elementwise kernel.

Generated kernels index with i32, so a tensor needs fewer than 2**31 elements. The lowering rejects what
it can prove too big (``index-range``) and records the rest in every spec as a precondition the host
checks at launch. Hand-written ``extern`` specs say nothing about indexing.
"""

import copy
import json

import pytest

import tilefish
from tilefish import Dim, Out, Tensor, TilefishLoweringError, TilefishSpecError, bf16, const, f32
from tilefish.aot import CompileSpec, compile_specs, declaration_digest

M = Dim("M")
N = Dim("N")
LIMIT = 2**31 - 1


def limits(spec):
    return [p for p in spec.preconditions if p.kind == "max_elements"]


def fails_index_range(op, **kwargs):
    with pytest.raises(TilefishLoweringError) as exc:
        op.lower(**kwargs)
    assert exc.value.rule == "index-range", str(exc.value)
    assert "rule 'index-range'" in str(exc.value) and str(LIMIT) in str(exc.value)
    return str(exc.value)


# -- static overflow is rejected ------------------------------------------------------------------------------------


def test_elementwise_kernel_with_too_many_static_elements_is_rejected():
    A, B = Dim("A", 70000), Dim("B", 70000)

    @tilefish.kernel
    def big(x: Tensor[A, B, f32]) -> Tensor[A, B, f32]:
        return x * 2.0

    text = fails_index_range(big, template="elementwise")
    assert "4900000000 elements" in text and "parameter 'x'" in text
    fails_index_range(big)  # whichever template would have been tried


def test_row_kernel_with_too_many_static_elements_is_rejected():
    A, D = Dim("A", 70000), Dim("D", 60000)

    @tilefish.kernel
    def rows(x: Tensor[A, D, f32]) -> Tensor[A, 1, f32]:
        return tilefish.sum(x, axis=-1, keepdims=True)

    fails_index_range(rows, template="row")
    fails_index_range(rows)


def test_the_limit_itself_is_allowed_and_one_more_is_not():
    @tilefish.kernel
    def at_limit(x: Tensor[LIMIT, f32]) -> Tensor[LIMIT, f32]:
        return x + 1.0

    (spec,) = compile_specs(at_limit.lower().extern)
    assert limits(spec) == []  # statically bounded, and within the limit

    @tilefish.kernel
    def over(x: Tensor[LIMIT + 1, f32]) -> Tensor[LIMIT + 1, f32]:
        return x + 1.0

    fails_index_range(over)


def test_a_static_factor_alone_can_make_any_row_count_overflow():
    wide = Dim("W", 3_000_000_000)

    @tilefish.kernel
    def k(x: Tensor[M, wide, f32]) -> Tensor[M, wide, f32]:  # M >= 1, so at least 3e9 elements
        return x + 1.0

    assert "at least 3000000000 elements" in fails_index_range(k)


def test_the_largest_bucket_counts():
    S = Dim("S", buckets=[8, 3_000_000_000])

    @tilefish.kernel
    def k(x: Tensor[S, 2, f32]) -> Tensor[S, 2, f32]:
        return x + 1.0

    assert "up to 6000000000 elements (its largest bucket)" in fails_index_range(k)

    ok = Dim("S2", buckets=[8, 1000])

    @tilefish.kernel
    def fine(x: Tensor[ok, 2, f32]) -> Tensor[ok, 2, f32]:
        return x + 1.0

    (spec,) = compile_specs(fine.lower().extern)
    assert limits(spec) == []  # every bucket is small: nothing for a host to check


def test_bucketed_row_length_is_bounded_by_its_largest_bucket():
    H = Dim("H", buckets=[256, 4096])
    R = Dim("R", 1_000_000)
    R2 = Dim("R2", 1000)

    @tilefish.kernel
    def k(x: Tensor[R, H, f32]) -> Tensor[R, 1, f32]:  # 1e6 * 4096 = 4.1e9 in the largest bucket
        return tilefish.sum(x, axis=-1, keepdims=True)

    fails_index_range(k)

    @tilefish.kernel
    def small(x: Tensor[R2, H, f32]) -> Tensor[R2, 1, f32]:
        return tilefish.sum(x, axis=-1, keepdims=True)

    specs = compile_specs(small.lower().extern)
    assert len(specs) == 2 and all(limits(s) == [] for s in specs)


def test_affine_and_scaled_shapes_use_interval_arithmetic():
    K = Dim("K", buckets=[1000, 3_000_000])

    @tilefish.kernel
    def scaled(x: Tensor[2 * K, 1000, f32]) -> Tensor[2 * K, 1000, f32]:  # 2 * 3e6 * 1e3 = 6e9
        return x + 1.0

    fails_index_range(scaled)

    @tilefish.kernel
    def halved(x: Tensor[K - 1, 100, f32]) -> Tensor[K - 1, 100, f32]:  # (3e6 - 1) * 100 < 2**31
        return x + 1.0

    assert all(limits(s) == [] for s in compile_specs(halved.lower().extern))


# -- dynamic sizes get a precondition -----------------------------------------------------------------------------


def test_softmax_over_a_big_vocabulary_has_a_precondition_for_every_tensor():
    V = Dim("V", 152064)

    @tilefish.kernel
    def softmax(x: Tensor[M, V, f32]) -> Tensor[M, V, f32]:
        e = tilefish.exp(x - tilefish.max(x, axis=-1, keepdims=True))
        return e / tilefish.sum(e, axis=-1, keepdims=True)

    lowered = softmax.lower()
    (spec,) = compile_specs(lowered.extern)
    assert [(p.arg, p.index, p.shape, p.value) for p in limits(spec)] == [
        ("x", 0, ("M", "152064"), LIMIT),
        ("out", 1, ("M", "152064"), LIMIT),
    ]
    assert limits(spec)[0].to_dict() == {
        "kind": "max_elements",
        "arg": "x",
        "index": 0,
        "value": LIMIT,
        "shape": ["M", "152064"],
    }
    assert lowered.extern.max_elements == {"x": LIMIT, "out": LIMIT}
    assert "int64" not in lowered.source  # indexing is still i32: the machine code does not change
    # it round-trips, and a stored copy is checked against the declaration
    back = CompileSpec.from_json(spec.to_json())
    assert back.to_dict() == spec.to_dict() and limits(back) == limits(spec)
    back.verify_against(lowered.extern)
    assert back.verified


def test_each_template_gets_preconditions_for_dynamic_shapes():
    @tilefish.kernel
    def ew(x: Tensor[M, 7, f32], y: Out[Tensor[M, 7, f32]]):
        y[...] = x * 2.0

    (spec,) = compile_specs(ew.lower().extern)
    assert [(p.arg, p.shape) for p in limits(spec)] == [("x", ("M", "7")), ("y", ("M", "7"))]

    @tilefish.kernel
    def rows(x: Tensor[M, 100, f32], w: Tensor[100, f32]) -> tuple[Tensor[M, 100, f32], Tensor[M, 1, f32]]:
        return x * w, tilefish.sum(x, axis=-1, keepdims=True)

    (spec,) = compile_specs(rows.lower().extern)
    # w is [100], statically small: no precondition; the outputs follow the rows
    assert [(p.arg, p.shape) for p in limits(spec)] == [("x", ("M", "100")), ("out0", ("M", "100")), ("out1", ("M", "1"))]


def test_a_statically_small_kernel_has_no_precondition():
    @tilefish.kernel
    def small(x: Tensor[64, 64, f32]) -> Tensor[64, 64, f32]:
        return x + 1.0

    lowered = small.lower()
    assert lowered.extern.max_elements == {}
    (spec,) = compile_specs(lowered.extern)
    assert limits(spec) == []


def test_bucketed_row_specs_state_the_bucket_value_in_the_shape():
    H = Dim("H", buckets=[256, 1024])

    @tilefish.kernel
    def rms(x: Tensor[M, H, f32]) -> Tensor[M, H, f32]:
        return x * tilefish.rsqrt(tilefish.mean(x * x, axis=-1, keepdims=True) + 1e-6)

    specs = compile_specs(rms.lower().extern)
    assert [[p.shape for p in limits(s)] for s in specs] == [[("M", "256")] * 2, [("M", "1024")] * 2]


def test_hand_written_externs_get_no_such_precondition():
    @tilefish.extern(triton="my_kernels:gelu", grid=lambda M: (M,))
    def ext(x: Tensor[M, 7, bf16], y: Out[Tensor[M, 7, bf16]], BLOCK: const = 1024): ...

    assert ext.max_elements == {} and ext.element_counts == {}
    (spec,) = compile_specs(ext)
    assert limits(spec) == [] and all("count_of" not in a.to_dict() for a in spec.args)


def test_the_facts_are_part_of_the_declaration_digest():
    @tilefish.kernel
    def k(x: Tensor[M, 7, f32]) -> Tensor[M, 7, f32]:
        return x * 2.0

    ext = k.lower().extern
    digests = {declaration_digest(ext)}
    for max_elements, element_counts in (({}, ext.element_counts), ({"x": LIMIT}, ext.element_counts), (ext.max_elements, {})):
        variant = tilefish.ExternOp(
            ext.fn, ext.name, ext.signature, ext.triton, ext.num_warps, ext.num_stages, ext.grid,
            max_elements=max_elements, element_counts=element_counts,
        )  # fmt: skip
        digests.add(declaration_digest(variant))
    assert len(digests) == 4  # every change of the facts changes the digest


def test_an_extern_validates_the_facts_it_is_given():
    @tilefish.kernel
    def k(x: Tensor[M, 7, f32]) -> Tensor[M, 7, f32]:
        return x * 2.0

    ext = k.lower().extern

    def build(**facts):
        return tilefish.ExternOp(
            ext.fn, ext.name, ext.signature, ext.triton, ext.num_warps, ext.num_stages, ext.grid, **facts
        )  # fmt: skip

    for bad in ({"nope": 1}, {"BLOCK": 1}, {"x": 0}, {"x": True}, {"x": "big"}):
        with pytest.raises(tilefish.TilefishDeclarationError, match="max_elements"):
            build(max_elements=bad)
    for bad in ({"nope": "x"}, {"x": "x"}, {"n": "nope"}, {"n": "BLOCK"}):
        with pytest.raises(tilefish.TilefishDeclarationError, match="element_counts"):
            build(element_counts=bad)


def test_a_spec_without_its_precondition_does_not_verify():
    @tilefish.kernel
    def k(x: Tensor[M, 7, f32]) -> Tensor[M, 7, f32]:
        return x * 2.0

    lowered = k.lower()
    (spec,) = compile_specs(lowered.extern)
    doc = spec.to_dict()
    doc["preconditions"] = [p for p in doc["preconditions"] if p["kind"] != "max_elements"]
    stripped = CompileSpec.from_dict(doc)  # well formed: only the declaration says it is incomplete
    with pytest.raises(TilefishSpecError, match="preconditions"):
        stripped.verify_against(lowered.extern)
    assert not stripped.verified


# -- the validator -------------------------------------------------------------------------------------------------


@pytest.fixture
def softmax_doc():
    V = Dim("V", 1000)

    @tilefish.kernel
    def softmax(x: Tensor[M, V, f32]) -> Tensor[M, V, f32]:
        e = tilefish.exp(x - tilefish.max(x, axis=-1, keepdims=True))
        return e / tilefish.sum(e, axis=-1, keepdims=True)

    (spec,) = compile_specs(softmax.lower().extern)
    doc = json.loads(spec.to_json())
    CompileSpec.from_dict(doc)  # valid as generated
    return doc


def limit_entries(doc):
    return [p for p in doc["preconditions"] if p["kind"] == "max_elements"]


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda p, doc: p.update(value=2**31), "must be 2147483647"),
        (lambda p, doc: p.update(value=2**63 - 1), "must be 2147483647"),
        (lambda p, doc: p.update(value=0), "must be an int >= 1"),
        (lambda p, doc: p.update(value="big"), "must be an int"),
        (lambda p, doc: p.update(shape=["M", "999"]), "own shape"),
        (lambda p, doc: p.update(shape="M*1000"), "list of dimension strings"),
        (lambda p, doc: p.update(shape=["M", "1000 +"]), "expected|unexpected|grammar"),
        (lambda p, doc: p.update(arg="nope"), "not an argument"),
        (lambda p, doc: p.update(index=1), "not an argument"),
        (lambda p, doc: p.pop("shape"), "missing"),
        (lambda p, doc: p.update(extra=1), "unknown"),
        (lambda p, doc: doc["preconditions"].append(dict(p)), "once"),
    ],
)
def test_malformed_max_elements_preconditions_are_rejected(softmax_doc, edit, message):
    doc = copy.deepcopy(softmax_doc)
    edit(limit_entries(doc)[0], doc)
    with pytest.raises(TilefishSpecError, match=message):
        CompileSpec.from_dict(doc)


def test_max_elements_must_sit_on_a_pointer_and_needs_schema_2(softmax_doc):
    doc = copy.deepcopy(softmax_doc)
    on_constexpr = next(a for a in doc["args"] if a["kind"] == "constexpr")
    limit_entries(doc)[0].update(arg=on_constexpr["name"], index=on_constexpr["index"])
    with pytest.raises(TilefishSpecError, match="pointer argument"):
        CompileSpec.from_dict(doc)

    old = copy.deepcopy(softmax_doc)
    old["schema"] = 1
    del old["declaration_digest"]
    old.pop("enable_fp_fusion", None)  # a schema 1 spec has neither, nor a source digest
    old.pop("source_sha256", None)
    with pytest.raises(TilefishSpecError, match="needs schema 2"):
        CompileSpec.from_dict(old)


def test_a_valid_spec_may_omit_a_precondition(softmax_doc):
    """The validator checks consistency, not completeness: the declaration is what says a precondition is due."""
    doc = copy.deepcopy(softmax_doc)
    doc["preconditions"].remove(limit_entries(doc)[1])
    CompileSpec.from_dict(doc)


# -- how a host computes the element count n of an elementwise kernel ------------------------------------------------


def count_arg(spec, name="n"):
    return next(a for a in spec.args if a.name == name)


def test_n_says_which_tensor_it_counts_when_the_product_is_affine():
    F = Dim("F", 3072)

    @tilefish.kernel
    def swiglu(gate: Tensor[M, F, bf16], up: Tensor[M, F, bf16]) -> Tensor[M, F, bf16]:
        return (tilefish.silu(gate.to(f32)) * up.to(f32)).to(bf16)

    lowered = swiglu.lower()
    (spec,) = compile_specs(lowered.extern)
    n = count_arg(spec)
    assert (n.dim, n.count_of) == ("3072*M", "gate")
    assert n.to_dict()["count_of"] == "gate"
    assert lowered.extern.element_counts == {"n": "gate"}
    assert CompileSpec.from_json(spec.to_json()).args[3].count_of == "gate"


def test_n_says_how_to_compute_it_when_the_product_is_not_affine():
    @tilefish.kernel
    def two(x: Tensor[M, N, f32], y: Tensor[M, N, f32]) -> Tensor[M, N, f32]:
        return x * y

    lowered = two.lower()
    (spec,) = compile_specs(lowered.extern)
    n = count_arg(spec)
    assert n.dim == "n" and n.count_of == "x"  # a fresh dimension, and the tensor a host takes its element count from
    assert [a.shape for a in spec.args if a.kind == "pointer"] == [("M", "N")] * 3
    assert "n = M * N" in lowered.source
    back = CompileSpec.from_json(spec.to_json())
    back.verify_against(lowered.extern)
    assert count_arg(back).count_of == "x"
    assert [(p.arg, p.shape) for p in limits(spec)] == [("x", ("M", "N")), ("y", ("M", "N")), ("out", ("M", "N"))]


def test_count_of_is_validated():
    @tilefish.kernel
    def two(x: Tensor[M, N, f32]) -> Tensor[M, N, f32]:
        return x * 2.0

    @tilefish.kernel
    def one(x: Tensor[M, 5, f32]) -> Tensor[M, 5, f32]:
        return x * 2.0

    (nonaffine,) = compile_specs(two.lower().extern)
    (affine,) = compile_specs(one.lower().extern)
    for spec in (nonaffine, affine):
        CompileSpec.from_json(spec.to_json())  # valid as generated

    def edited(spec, **changes):
        doc = json.loads(spec.to_json())
        next(a for a in doc["args"] if a["name"] == "n").update(changes)
        return doc

    # it must name a pointer argument
    for bad in ("nope", "n", "BLOCK"):
        with pytest.raises(TilefishSpecError, match="count_of"):
            CompileSpec.from_dict(edited(nonaffine, count_of=bad))
    with pytest.raises(TilefishSpecError, match="identifier"):
        CompileSpec.from_dict(edited(nonaffine, count_of="a b"))
    # an affine product must be what the scalar carries
    with pytest.raises(TilefishSpecError, match="carries"):
        CompileSpec.from_dict(edited(affine, dim="6*M"))
    with pytest.raises(TilefishSpecError, match="carries"):
        CompileSpec.from_dict(edited(affine, count_of="out", dim="M"))
    # a non-affine product needs a fresh dimension name, not an expression that pretends to be the count
    with pytest.raises(TilefishSpecError, match="fresh dimension name"):
        CompileSpec.from_dict(edited(nonaffine, dim="M + N"))
    # only scalars carry it, and only from schema 2 on
    doc = json.loads(nonaffine.to_json())
    next(a for a in doc["args"] if a["kind"] == "pointer")["count_of"] = "x"
    with pytest.raises(TilefishSpecError, match="must have exactly the fields"):
        CompileSpec.from_dict(doc)
    old = json.loads(nonaffine.to_json())
    old["schema"] = 1
    del old["declaration_digest"]
    old.pop("enable_fp_fusion", None)  # a schema 1 spec has neither, nor a source digest
    old.pop("source_sha256", None)
    old["preconditions"] = [p for p in old["preconditions"] if p["kind"] != "max_elements"]
    with pytest.raises(TilefishSpecError, match="must have exactly the fields"):
        CompileSpec.from_dict(old)  # schema 1 has no count_of


def test_a_changed_count_of_does_not_verify():
    @tilefish.kernel
    def two(x: Tensor[M, N, f32], y: Tensor[M, N, f32]) -> Tensor[M, N, f32]:
        return x * y

    lowered = two.lower()
    (spec,) = compile_specs(lowered.extern)
    doc = json.loads(spec.to_json())
    next(a for a in doc["args"] if a["name"] == "n")["count_of"] = "y"
    other = CompileSpec.from_dict(doc)  # as valid as the original: y has the same shape
    with pytest.raises(TilefishSpecError, match="count_of"):
        other.verify_against(lowered.extern)
