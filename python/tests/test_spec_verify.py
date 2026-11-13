"""CompileSpec.verify_against, declaration_digest, constexpr dtypes and schema 2.

A serialized spec that is self-consistent validates in ``from_dict`` even when it is not the spec of the
declaration it claims to come from; ``verify_against`` regenerates the spec and requires an exact match.
"""

import copy
import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import tilefish
from tilefish import Dim, InOut, Out, Tensor, TilefishSpecError, bf16, const, f32, i32
from tilefish.aot import CompileSpec, compile_specs, declaration_digest

S = Dim("S", buckets=[16, 32])
n = Dim("n", divisible_by=16)


@tilefish.extern(triton="my_kernels.sub:fn", num_warps=4, num_stages=3, grid=lambda S, n: (tilefish.cdiv(S, 4), n))
def kernel(
    x: Tensor[S, n, bf16],
    w: InOut[Tensor[3, f32]],
    y: Out[Tensor[S, n, bf16]],
    cnt: n,
    odd: S,
    *,
    B: const = S,
    E: const[f32] = 1.0,
    N: const[f32] = -2.5,
    Z: const[f32] = 0.0,
    I: const[i32] = 7,
    T: const = True,
): ...


SPECS = compile_specs(kernel)
SPEC = SPECS[0]
INDEX = {a.name: a.index for a in SPEC.args}

# The digest of `kernel`, pinned: it must not change between runs, platforms or Python versions. If this
# fails on purpose (the canonical form changed), bump aot.DIGEST_FORMAT and update the literal.
KERNEL_DIGEST = "31f1efc428e79da02111c830cbfca7486095148b7677c2e185acf097355fe198"


def forged(mutate, spec=SPEC):
    """Edit a serialized spec and read it back. The edit must be self-consistent: from_dict accepts it."""
    doc = copy.deepcopy(spec.to_dict())
    mutate(doc)
    return CompileSpec.from_dict(doc)


def assert_caught(spec, field, op=kernel):
    with pytest.raises(TilefishSpecError) as info:
        spec.verify_against(op)
    text = str(info.value)
    assert f"first difference at {field}:" in text, text
    assert f"compile spec '{spec.stem}'" in text and f"extern op '{op.name}'" in text
    return text


def set_const(name, value, index=INDEX):
    def mutate(doc):
        doc["constants"][str(index[name])] = value
        doc["args"][index[name]]["value"] = value

    return mutate


# -- generated specs verify ---------------------------------------------------------------------------------------------------------


def test_generated_specs_verify_and_survive_a_round_trip():
    assert len(SPECS) == 2
    for spec in SPECS:
        assert spec.verifiable and spec.schema == 2
        assert spec.verify_against(kernel) is None
        assert CompileSpec.from_dict(spec.to_dict()).verify_against(kernel) is None
        assert CompileSpec.from_json(spec.to_json()).verify_against(kernel) is None
        assert CompileSpec.from_json(spec.to_json(indent=None)).verify_against(kernel) is None


def test_non_finite_constants_verify_after_a_json_round_trip_and_edits_are_caught():
    @tilefish.extern(triton="m:f", grid=(1,))
    def special(x: Tensor[n, bf16], A: const[f32] = float("inf"), B: const[f32] = float("nan"), C: const = float("-inf")): ...

    (spec,) = compile_specs(special)
    back = CompileSpec.from_json(spec.to_json())
    assert back.verify_against(special) is None  # nan equals itself here
    index = {a.name: a.index for a in spec.args}
    assert_caught(forged(set_const("B", {"float": "inf"}, index), back), f"args[{index['B']}].value.float", special)
    assert_caught(forged(set_const("A", {"float": "-inf"}, index), back), f"args[{index['A']}].value.float", special)
    assert_caught(forged(set_const("C", 5, index), back), f"args[{index['C']}].value", special)


def test_verify_against_needs_an_extern_op():
    with pytest.raises(TilefishSpecError, match="takes an extern op"):
        SPEC.verify_against(lambda: None)  # type: ignore[arg-type]


# -- self-consistent forgeries: from_dict accepts them, verify_against does not ---------------------------------------------------------


def test_changing_a_constexpr_in_constants_and_args_alike_is_caught():
    spec = forged(set_const("B", 17))
    assert spec.constants[(INDEX["B"],)] == 17  # it validated: from_dict could not tell
    text = assert_caught(spec, f"args[{INDEX['B']}].value")
    assert "the spec has 17, the declaration gives 16" in text


@pytest.mark.parametrize(
    "name,value",
    [
        ("E", 1),  # float -> int
        ("E", True),  # float -> bool
        ("E", 2.0),
        ("N", -0.0),
        ("Z", -0.0),  # 0.0 == -0.0 in python, but they are different constants
        ("Z", 0),  # 0.0 == 0
        ("Z", False),  # 0.0 == False
        ("I", 7.0),  # 7 == 7.0
        ("I", 8),
        ("T", 1),  # True == 1
        ("T", False),
    ],
)
def test_a_constexpr_of_another_value_or_python_type_is_caught(name, value):
    assert_caught(forged(set_const(name, value)), f"args[{INDEX[name]}].value")


def test_pointer_dtype_changed_in_every_field_is_caught():
    def mutate(doc):
        a = doc["args"][INDEX["x"]]
        a["type"], a["dtype"] = "*fp32", "f32"
        doc["signature"]["x"] = "*fp32"
        doc["launch_params"][0]["type"] = "*fp32"

    text = assert_caught(forged(mutate), f"args[{INDEX['x']}].type")
    assert "'*fp32'" in text and "'*bf16'" in text


@pytest.mark.parametrize(
    "mutate,field",
    [
        (lambda d: d.update(num_warps=8, block=[256, 1, 1]), "num_warps"),
        (lambda d: d.update(num_warps=2, block=[64, 1, 1]), "num_warps"),
        (lambda d: d.update(num_stages=2), "num_stages"),
        (lambda d: d.update(grid="(S, n)"), "grid"),
        (lambda d: d.update(grid="(4, n, 1)"), "grid"),
        (lambda d: d["triton"].update(module="evil", ref="evil:fn"), "triton.module"),
        (lambda d: d["triton"].update(function="other", ref="my_kernels.sub:other"), "triton.function"),
        (lambda d: d["args"][0].update(access="out"), "args[0].access"),
        (lambda d: d["args"][1].update(access="in"), "args[1].access"),
        (lambda d: d.update(declaration_digest="0" * 64), "declaration_digest"),
        (lambda d: d["args"][INDEX["E"]].pop("dtype"), f"args[{INDEX['E']}].dtype"),
        (lambda d: d["args"][INDEX["E"]].update(dtype="bf16"), f"args[{INDEX['E']}].dtype"),
        (lambda d: d["args"][INDEX["B"]].update(dtype="i32"), f"args[{INDEX['B']}].dtype"),
    ],
)
def test_other_self_consistent_edits_are_caught_and_name_the_field(mutate, field):
    assert_caught(forged(mutate), field)


def test_a_static_size_edited_together_with_its_shape_is_caught():
    def mutate(doc):
        doc["args"][INDEX["w"]]["shape"] = ["4"]
        doc["args"][INDEX["w"]]["shape_dims"] = [4]
        for p in doc["preconditions"]:
            if p["kind"] == "static_size" and p["arg"] == "w":
                p["value"] = 4
            if p["kind"] == "max_elements" and p["arg"] == "w":
                p["shape"] = ["4"]

    assert_caught(forged(mutate), f"args[{INDEX['w']}].shape[0]")


def test_dropping_a_divisibility_hint_together_with_its_precondition_is_caught():
    def mutate(doc):
        del doc["attrs"][str(INDEX["cnt"])]
        doc["preconditions"] = [p for p in doc["preconditions"] if p["kind"] != "divisible"]

    assert_caught(forged(mutate), f"attrs.{INDEX['cnt']}")


def test_a_bucket_spec_relabelled_as_another_bucket_is_caught():
    def mutate(doc):
        doc["buckets"] = {"S": 32}
        doc["stem"] = "kernel.S32"
        doc["dims"]["S"]["bucket"] = 32
        for p in doc["preconditions"]:
            if p["kind"] == "bucket":
                p["value"] = 32

    assert_caught(forged(mutate), "args[0].shape[0]")  # still the shapes of the S=16 binary


def test_a_spec_of_a_binary_or_operator_the_declaration_does_not_generate_is_caught():
    def to_other_bucket(doc):
        doc["buckets"] = {"S": 64}
        doc["stem"] = "kernel.S64"
        doc["dims"]["S"].update(bucket=64, buckets=[16, 32, 64])
        for p in doc["preconditions"]:
            if p["kind"] == "bucket":
                p.update(value=64, buckets=[16, 32, 64])

    text = assert_caught(forged(to_other_bucket), "stem")
    assert "kernel.S16" in text and "kernel.S32" in text  # the binaries the declaration does generate

    @tilefish.extern(triton="m:f", grid=(1,))
    def other(x: Tensor[n, bf16]): ...

    (other_spec,) = compile_specs(other)
    assert_caught(other_spec, "op")


def test_edits_made_without_from_dict_are_caught_too():
    assert_caught(dataclasses.replace(SPEC, constants={**SPEC.constants, (INDEX["B"],): 17}), f"constants.{INDEX['B']}")
    assert_caught(dataclasses.replace(SPEC, num_stages=9), "num_stages")
    assert_caught(dataclasses.replace(SPEC, preconditions=SPEC.preconditions[:-1]), "preconditions (length)")
    odd = {**SPEC.attrs, (INDEX["odd"],): [["tt.divisibility", 16]]}
    assert_caught(dataclasses.replace(SPEC, attrs=odd), f"attrs.{INDEX['odd']}")
    assert_caught(dataclasses.replace(SPEC, signature=dict(reversed(list(SPEC.signature.items())))), "signature (order)")
    assert_caught(dataclasses.replace(SPEC, declaration_digest="f" * 64), "declaration_digest")
    assert_caught(dataclasses.replace(SPEC, block=(256, 1, 1)), "block[0]")


def test_a_changed_declaration_no_longer_verifies_the_old_spec():
    spec = compile_specs(_declare())[0]
    spec.verify_against(_declare())
    assert_caught(spec, "args[2].value", _declare(value=32))
    assert_caught(spec, "args[0].type", _declare(dtype=f32))
    assert_caught(spec, "num_warps", _declare(warps=8))
    assert_caught(spec, "triton.module", _declare(triton="m2:f"))
    assert_caught(spec, "grid", _declare(grid=lambda n: (n, 2)))
    assert_caught(spec, "args[2].dtype", _declare(value=16, cdtype=i32))


# -- the declaration digest ---------------------------------------------------------------------------------------------------------


def _declare(
    *,
    name="e",
    triton="m:f",
    warps=4,
    stages=3,
    grid=lambda n: (n,),
    dtype=bf16,
    shape=None,
    access="in",
    dim=n,
    value=16,
    cdtype=None,
    extra=False,
    swap=False,
):
    """One small extern; every keyword changes one thing about its declaration."""
    ty = Tensor[(*(shape if shape is not None else (dim,)), dtype)]
    xt = {"in": ty, "inout": InOut[ty], "out": Out[ty]}[access]
    ann = const[cdtype] if cdtype is not None else const

    if swap:

        def e(c: dim, x: xt, *, BLOCK: ann = value): ...

    elif extra:

        def e(x: xt, c: dim, d: dim, *, BLOCK: ann = value): ...

    else:

        def e(x: xt, c: dim, *, BLOCK: ann = value): ...

    return tilefish.extern(triton=triton, num_warps=warps, num_stages=stages, grid=grid, name=name)(e)


def test_the_digest_is_a_pinned_stable_sha256():
    digest = declaration_digest(kernel)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert digest == KERNEL_DIGEST
    assert {s.declaration_digest for s in SPECS} == {digest}  # the same for every bucket of the op
    assert SPEC.to_dict()["declaration_digest"] == digest
    assert list(SPEC.to_dict())[-1] == "declaration_digest"


def test_the_digest_does_not_depend_on_the_hash_seed_or_the_process():
    code = (
        "import tilefish\n"
        "from tilefish import Dim, Tensor, bf16, const, f32\n"
        "from tilefish.aot import declaration_digest\n"
        "S = Dim('S', buckets=[32, 16]); n = Dim('n', divisible_by=16)\n"
        "@tilefish.extern(triton='m:f', grid=lambda S, n: (tilefish.cdiv(S, 4), n))\n"
        "def e(x: Tensor[S, n, bf16], c: n, *, B: const = tilefish.cdiv(S, 2), E: const[f32] = float('nan')): ...\n"
        "print(declaration_digest(e))\n"
    )
    root = str(Path(tilefish.__file__).resolve().parent.parent)
    seen = set()
    for seed in ("0", "1", "424242", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": root}
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        seen.add(out.stdout.strip())
    assert len(seen) == 1 and re.fullmatch(r"[0-9a-f]{64}", seen.pop())


def test_the_digest_changes_with_every_field_that_reaches_a_spec():
    base = declaration_digest(_declare())
    variants = {
        "name": _declare(name="e2"),
        "triton module": _declare(triton="m2:f"),
        "triton function": _declare(triton="m:g"),
        "num_warps": _declare(warps=8),
        "num_stages": _declare(stages=2),
        "grid": _declare(grid=lambda n: (n, 2)),
        "grid expression": _declare(grid=lambda n: (tilefish.cdiv(n, 4),)),
        "pointer dtype": _declare(dtype=f32),
        "shape": _declare(shape=(n, 4)),
        "access inout": _declare(access="inout"),
        "divisible_by": _declare(dim=Dim("n", divisible_by=32)),
        "no divisibility": _declare(dim=Dim("n")),
        "buckets": _declare(dim=Dim("n", divisible_by=16, buckets=[16, 32])),
        "other buckets": _declare(dim=Dim("n", divisible_by=16, buckets=[16, 64])),
        "dim name": _declare(dim=Dim("m", divisible_by=16), grid=lambda m: (m,)),
        "constexpr value": _declare(value=17),
        "constexpr type float": _declare(value=16.0),
        "constexpr type bool": _declare(value=True),
        "constexpr dtype f32": _declare(value=16.0, cdtype=f32),
        "constexpr dtype i32": _declare(value=16, cdtype=i32),
        "extra parameter": _declare(extra=True),
        "parameter order": _declare(swap=True),
    }
    digests = {k: declaration_digest(v) for k, v in variants.items()}
    assert base not in digests.values()
    assert len(set(digests.values())) == len(digests), digests
    assert declaration_digest(_declare()) == base  # and it is repeatable


def test_constexprs_that_compare_equal_in_python_digest_differently():
    def digest(ann, value):
        @tilefish.extern(triton="m:f", grid=(1,))
        def e(x: Tensor[n, bf16], *, C: ann = value): ...

        return declaration_digest(e)

    forms = [
        (const, 1),
        (const, 1.0),
        (const, True),
        (const[f32], 1.0),
        (const[i32], 1),
        (const[tilefish.i64], 1),
        (const, 0.0),
        (const, -0.0),
        (const, float("nan")),
        (const, float("inf")),
        (const, float("-inf")),
    ]
    assert len({digest(a, v) for a, v in forms}) == len(forms)


def test_what_no_spec_can_show_does_not_change_the_digest():
    D = Dim("D", 2560)

    def with_names(x: Tensor[n, D, bf16], c: n, *, B: const = D, BLOCK: const = 128):
        """Docs."""

    def with_literals(x: Tensor[n, 2560, bf16], c: n, B: const = 2560, BLOCK: const = 128):
        pass  # no keyword-only marker, a literal where the other names a static Dim

    a = tilefish.extern(triton="m:f", grid=lambda n, BLOCK: (tilefish.cdiv(n, BLOCK), 1), name="op")(with_names)
    b = tilefish.extern(triton="m:f", grid=(tilefish.cdiv(n, 128), 1), name="op")(with_literals)
    assert compile_specs(a)[0].to_dict() == compile_specs(b)[0].to_dict()
    assert declaration_digest(a) == declaration_digest(b)


def test_declaration_digest_takes_an_extern_op():
    with pytest.raises(TilefishSpecError, match="takes an extern op"):
        declaration_digest(len)  # type: ignore[arg-type]


# -- constexpr dtypes -------------------------------------------------------------------------------------------------------------


def test_constexpr_dtypes_are_recorded_and_survive_a_round_trip():
    @tilefish.extern(triton="m:f", grid=(1,))
    def typed(
        x: Tensor[n, bf16],
        *,
        A: const[f32] = 1.0,
        B: const = 1.0,
        C: const = 1,
        D: const[i32] = 1,
        E: const = True,
        F: const[f32] = float("-inf"),
    ): ...

    (spec,) = compile_specs(typed)
    assert [(a.name, a.dtype, type(a.value)) for a in spec.args[1:]] == [
        ("A", "f32", float),
        ("B", None, float),
        ("C", None, int),
        ("D", "i32", int),
        ("E", None, bool),
        ("F", "f32", float),
    ]
    doc = json.loads(spec.to_json())
    assert doc["args"][1] == {"index": 1, "name": "A", "kind": "constexpr", "type": "constexpr", "dtype": "f32", "value": 1.0}
    assert doc["args"][2] == {"index": 2, "name": "B", "kind": "constexpr", "type": "constexpr", "value": 1.0}  # untyped: unchanged
    assert doc["args"][3] == {"index": 3, "name": "C", "kind": "constexpr", "type": "constexpr", "value": 1}
    back = CompileSpec.from_json(spec.to_json())
    assert back == spec and back.verify_against(typed) is None
    assert [(a.dtype, type(a.value)) for a in back.args[1:]] == [(a.dtype, type(a.value)) for a in spec.args[1:]]
    # `const[f32] = 1.0` is told apart from an int `1` and from an untyped `1.0`
    assert (back.args[1].dtype, back.args[1].value) != (back.args[3].dtype, back.args[3].value)
    assert back.args[1].dtype != back.args[2].dtype
    assert back.args[1].value == back.args[2].value and type(back.args[1].value) is type(back.args[2].value)


def test_constexpr_dtype_must_be_a_known_dtype_name():
    for bad in ("f64", "", 5, None, ["f32"], "F32"):
        doc = copy.deepcopy(SPEC.to_dict())
        doc["args"][INDEX["E"]]["dtype"] = bad
        with pytest.raises(TilefishSpecError, match="constexpr's dtype must be one of"):
            CompileSpec.from_dict(doc)


# -- schema 2 and reading schema 1 ---------------------------------------------------------------------------------------------------


def as_schema_1(spec):
    """What schema 1 wrote for a spec: no digest and no constexpr dtypes."""
    doc = copy.deepcopy(spec.to_dict())
    doc["schema"] = 1
    del doc["declaration_digest"]
    doc.pop("dims", None)  # schema 1 has no table of dimensions, and no declared shapes
    doc["preconditions"] = [p for p in doc["preconditions"] if p["kind"] != "max_elements"]  # nor max_elements
    for a in doc["args"]:
        a.pop("shape_dims", None)
        if a["kind"] == "constexpr":
            a.pop("dtype", None)
    return doc


def test_the_schema_is_2_and_the_digest_is_part_of_the_document():
    doc = SPEC.to_dict()
    assert doc["schema"] == 2 and SPEC.schema == 2
    assert re.fullmatch(r"[0-9a-f]{64}", doc["declaration_digest"])
    assert json.loads(SPEC.to_json())["declaration_digest"] == doc["declaration_digest"]
    assert CompileSpec.from_dict(doc).declaration_digest == doc["declaration_digest"]  # from_dict keeps it


def test_schema_1_specs_are_still_read_but_cannot_be_verified():
    doc = as_schema_1(SPEC)
    old = CompileSpec.from_dict(doc)
    assert old.schema == 1 and not old.verifiable and old.declaration_digest is None
    assert old.signature == SPEC.signature and old.constants == SPEC.constants and old.attrs == SPEC.attrs
    assert all(a.dtype is None for a in old.args if a.kind == "constexpr")
    assert not old.verified
    with pytest.raises(TilefishSpecError, match=r"schema 1.*cannot be verified.*regenerate it with compile_specs") as info:
        old.verify_against(kernel)
    assert "kernel.S16" in str(info.value)
    # and it cannot feed a compile either: there is nothing to verify it with
    with pytest.raises(TilefishSpecError, match=r"schema 1 and has not been verified.*regenerate it with compile_specs"):
        old.ast_source_inputs()
    # it is written back as schema 1, not silently upgraded to a schema 2 it cannot back up
    assert old.to_dict() == doc
    assert CompileSpec.from_json(old.to_json()) == old
    # regenerating gives a verifiable spec for the same binary
    fresh = {s.stem: s for s in compile_specs(kernel)}[old.stem]
    fresh.verify_against(kernel)
    assert fresh.verified and fresh.ast_source_inputs().constants == old.constants


def test_every_schema_1_document_of_every_binary_is_read():
    for spec in SPECS:
        old = CompileSpec.from_json(json.dumps(as_schema_1(spec)))
        assert not old.verifiable and old.stem == spec.stem


def test_each_schema_is_strict_about_its_own_fields():
    doc = copy.deepcopy(SPEC.to_dict())
    del doc["declaration_digest"]
    with pytest.raises(TilefishSpecError, match="is missing"):  # schema 2 needs the digest
        CompileSpec.from_dict(doc)

    old = as_schema_1(SPEC)
    with_digest = {**old, "declaration_digest": SPEC.declaration_digest}
    with pytest.raises(TilefishSpecError, match="unknown fields"):  # schema 1 has none
        CompileSpec.from_dict(with_digest)
    with_dtype = copy.deepcopy(old)
    with_dtype["args"][INDEX["E"]]["dtype"] = "f32"
    with pytest.raises(TilefishSpecError, match="must have exactly the fields"):  # nor constexpr dtypes
        CompileSpec.from_dict(with_dtype)

    for schema in (0, 3, 99, -1, None, "2", 2.0, 1.0, True, False, [2]):
        doc = copy.deepcopy(SPEC.to_dict())
        doc["schema"] = schema
        with pytest.raises(TilefishSpecError, match="unsupported compile spec schema"):
            CompileSpec.from_dict(doc)


@pytest.mark.parametrize(
    "digest", ["", "0" * 63, "0" * 65, "A" * 64, "g" * 64, " " + "0" * 63, "0" * 64 + "\n", None, 5, ["0" * 64]]
)
def test_the_digest_must_be_64_lowercase_hex_digits(digest):
    doc = copy.deepcopy(SPEC.to_dict())
    doc["declaration_digest"] = digest
    with pytest.raises(TilefishSpecError, match="declaration_digest must be 64 lowercase hex digits"):
        CompileSpec.from_dict(doc)


# -- what compile_specs emits for a valid extern is unchanged except for the additive fields ------------------------------------------


def test_schema_2_only_adds_the_digest_the_constexpr_dtypes_and_what_a_host_binds_dimensions_by_to_schema_1():
    for spec in SPECS:
        new, old = spec.to_dict(), as_schema_1(spec)
        assert set(new) - set(old) == {"declaration_digest", "dims"}
        for key in old:
            if key == "preconditions":  # and a max_elements precondition per tensor that indexes with i32
                assert [p for p in new[key] if p["kind"] != "max_elements"] == old[key], key
                assert {p["arg"] for p in new[key] if p["kind"] == "max_elements"} == {a["name"] for a in new["args"] if a["kind"] == "pointer"}
            elif key not in ("schema", "args"):
                assert new[key] == old[key], key
        for a_new, a_old in zip(new["args"], old["args"]):
            added = {k: v for k, v in a_new.items() if not (a_new["kind"] == "constexpr" and k == "dtype") and k != "shape_dims"}
            assert added == a_old
