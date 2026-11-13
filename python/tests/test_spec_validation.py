"""CompileSpec.from_dict / from_json treat their input as untrusted: strict, consistent, never evaluated."""

import copy
import json

import pytest

import tilefish
from tilefish import Dim, InOut, Out, Tensor, TilefishSpecError, bf16, const, f32
from tilefish._specvalid import parse_expression, parse_grid
from tilefish.aot import ArgSpec, CompileSpec, LaunchParam, Precondition, compile_specs

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
    E: const[f32] = float("inf"),
    N: const[f32] = -2.5,
): ...


(SPEC, *_) = compile_specs(kernel)
GOOD = SPEC.to_dict()
INDEX = {a["name"]: a["index"] for a in GOOD["args"]}


def tampered(mutate):
    d = copy.deepcopy(GOOD)
    mutate(d)
    return d


def rejects(mutate, match=None):
    with pytest.raises(TilefishSpecError, match=match):
        CompileSpec.from_dict(tampered(mutate))


def test_the_untouched_spec_is_accepted_and_every_generated_spec_validates():
    assert CompileSpec.from_dict(copy.deepcopy(GOOD)) == SPEC
    for spec in compile_specs(kernel):
        assert CompileSpec.from_dict(spec.to_dict()) == spec
    assert GOOD["args"][INDEX["cnt"]]["dim"] == "n" and GOOD["args"][INDEX["odd"]]["dim"] == "S"


# -- signature <=> args ------------------------------------------------------------------------------------------------------------


def test_signature_must_agree_with_the_args():
    rejects(lambda d: d["signature"].update({"x": "*fp32"}), "signature must list exactly")
    rejects(lambda d: d.update(signature=dict(reversed(list(d["signature"].items())))), "in order")
    rejects(lambda d: d["signature"].pop("cnt"), "signature must list exactly")
    rejects(lambda d: d["signature"].update({"extra": "i32"}), "signature must list exactly")
    rejects(lambda d: d.update(signature={("renamed" if k == "x" else k): v for k, v in d["signature"].items()}), "signature")
    rejects(lambda d: d["signature"].update({"cnt": "constexpr"}), "signature must list exactly")
    rejects(lambda d: d.update(signature=[1, 2]), "signature must list exactly")


def test_args_must_be_consistent_with_their_kind():
    rejects(lambda d: d["args"][INDEX["x"]].update(kind="scalar"), "must have exactly the fields")
    rejects(lambda d: d["args"][INDEX["cnt"]].update(kind="pointer"), "must have exactly the fields")
    rejects(lambda d: d["args"][INDEX["x"]].update(type="*i32"), "does not match dtype")
    rejects(lambda d: d["args"][INDEX["x"]].update(type="*nonsense", dtype="bf16"), "does not match dtype")
    rejects(lambda d: d["args"][INDEX["cnt"]].update(type="i64", dtype="i64"), "must be i32")
    rejects(lambda d: d["args"][INDEX["B"]].update(type="i32"), "must be 'constexpr'")
    rejects(lambda d: d["args"][INDEX["x"]].update(access="readwrite"), "access must be in, inout or out")
    rejects(lambda d: d["args"][INDEX["x"]].update(index=7), "has index 7 but is at position")
    rejects(lambda d: d["args"][INDEX["y"]].update(name="x"), "names must be unique")
    rejects(lambda d: d["args"][INDEX["y"]].update(name="not an identifier"), "identifier")
    rejects(lambda d: d["args"][INDEX["x"]].update(shape=["S", "__import__('os')"]), "dimension grammar")
    rejects(lambda d: d["args"][INDEX["cnt"]].update(dim="n; drop()"), "dimension grammar")
    rejects(lambda d: d["args"][INDEX["x"]].update(shape="S,n"), "shape must be a list")
    rejects(lambda d: d["args"][INDEX["x"]].update(extra=1), "unknown fields|must have exactly the fields")


# -- constants only at constexpr indices -------------------------------------------------------------------------------------------


def test_constants_sit_at_the_constexpr_indices_and_every_constexpr_has_one():
    rejects(lambda d: d["constants"].update({str(INDEX["x"]): 5}), "exactly one entry per constexpr")
    rejects(lambda d: d["constants"].pop(str(INDEX["B"])), "exactly one entry per constexpr")
    rejects(lambda d: d["constants"].update({str(INDEX["cnt"]): 1}), "exactly one entry per constexpr")
    rejects(lambda d: d["constants"].update({str(INDEX["B"]): 17}), "differs from the value of arg B")
    rejects(lambda d: d["constants"].update({str(INDEX["B"]): 16.0}), "differs from the value")
    rejects(lambda d: d["constants"].update({str(INDEX["B"]): True}), "differs from the value")
    rejects(lambda d: d["constants"].update({"x": 1}), "must be an argument index")
    rejects(lambda d: d["constants"].update({"-1": 1}), "must be an argument index")
    rejects(lambda d: d.update(constants=[]), "constants must be an object")
    rejects(lambda d: d["constants"].update({str(INDEX["E"]): {"float": "foo"}}), "not a valid tagged float")
    rejects(lambda d: d["constants"].update({str(INDEX["E"]): {"float": "1e999"}}), "not a valid tagged float")
    rejects(lambda d: d["constants"].update({str(INDEX["E"]): {"float": "inf", "x": 1}}), "not a valid tagged float")
    rejects(lambda d: d["constants"].update({str(INDEX["E"]): "inf"}), "must be a number or a bool")
    rejects(lambda d: d["constants"].update({str(INDEX["E"]): [1]}), "must be a number or a bool")
    rejects(lambda d: d["args"][INDEX["E"]].update(value=float("inf")), "non-finite float outside a tag")


# -- attrs ------------------------------------------------------------------------------------------------------------------------


def test_attrs_are_exactly_the_pointers_and_the_scalars_with_a_divisibility_precondition():
    rejects(lambda d: d["attrs"].update({str(INDEX["odd"]): [["tt.divisibility", 16]]}), "attrs must be at exactly")  # no precondition
    rejects(lambda d: d["attrs"].update({str(INDEX["B"]): [["tt.divisibility", 16]]}), "attrs must be at exactly")  # constexpr
    rejects(lambda d: d["attrs"].pop(str(INDEX["x"])), "attrs must be at exactly")  # a pointer without its hint
    rejects(lambda d: d["attrs"].pop(str(INDEX["cnt"])), "attrs must be at exactly")  # divisible scalar without it
    rejects(lambda d: d["attrs"].update({"99": [["tt.divisibility", 16]]}), "in range")
    rejects(lambda d: d["attrs"].update({"x": [["tt.divisibility", 16]]}), "argument index")


def test_attrs_values_are_exactly_a_16_byte_divisibility_hint():
    key = str(INDEX["x"])
    for bad in (
        [["tt.divisibility", 1024]],
        [["tt.divisibility", 8]],
        [["tt.divisibility", 16], ["tt.divisibility", 16]],
        [["tt.max_divisibility", 16]],
        [["tt.divisibility", "16"]],
        [["tt.divisibility", True]],
        [["tt.divisibility", 16.0]],  # equal to 16, but not an int
        [["tt.divisibility"]],
        [["tt.divisibility", 16, 1]],
        [[16, "tt.divisibility"]],
        [],
        "tt.divisibility",
        None,
    ):
        rejects(lambda d, bad=bad: d["attrs"].update({key: bad}), "must be exactly")


def test_a_divisible_precondition_must_be_a_multiple_of_16():
    pre = next(i for i, p in enumerate(GOOD["preconditions"]) if p["kind"] == "divisible")
    rejects(lambda d: d["preconditions"][pre].update(by=8), "multiple of 16")
    rejects(lambda d: d["preconditions"][pre].update(dim="S"), "its own dim")
    rejects(lambda d: d["preconditions"][pre].update(arg="x"), "not an argument|divisible must be")
    ok = tampered(lambda d: d["preconditions"][pre].update(by=32))
    assert CompileSpec.from_dict(ok).preconditions[pre].by == 32


# -- launch settings ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("warps", [0, 3, 5, 64, -4, True, 4.0, "4", None])
def test_num_warps_is_a_power_of_two_up_to_32(warps):
    rejects(lambda d: d.update(num_warps=warps), "num_warps must be one of")


@pytest.mark.parametrize("warps", [1, 2, 8, 16, 32])
def test_valid_num_warps_need_a_matching_block(warps):
    rejects(lambda d: d.update(num_warps=warps), "block must be")
    ok = tampered(lambda d: d.update(num_warps=warps, block=[32 * warps, 1, 1]))
    assert CompileSpec.from_dict(ok).block == (32 * warps, 1, 1)


def test_block_follows_num_warps():
    for bad in ([7, 1, 1], [128, 1], [128, 2, 1], [128.0, 1, 1], "128", None, [128, 1, True]):
        rejects(lambda d, bad=bad: d.update(block=bad), "block must be")


def test_num_stages_is_a_small_positive_int():
    for bad in (0, -1, 17, 1000, True, 2.0, "3", None):
        rejects(lambda d, bad=bad: d.update(num_stages=bad), "num_stages must be")


def test_launch_params_are_the_non_constexpr_args_plus_the_two_zero_scratch_slots():
    rejects(lambda d: d["launch_params"].pop(), "launch_params must be")
    rejects(lambda d: d["launch_params"].__delitem__(slice(-2, None)), "launch_params must be")
    rejects(lambda d: d["launch_params"][-1].update(value=1), "launch_params must be")
    rejects(lambda d: d["launch_params"][-2].update(type="i64"), "launch_params must be")
    rejects(lambda d: d["launch_params"].insert(0, {"name": "B", "type": "constexpr"}), "launch_params must be")
    rejects(lambda d: d["launch_params"].reverse(), "launch_params must be")
    rejects(lambda d: d["launch_params"].append({"name": "extra", "type": "u64", "value": 0}), "launch_params must be")
    rejects(lambda d: d["launch_params"][0].update(name="renamed"), "launch_params must be")
    rejects(lambda d: d["launch_params"][0].update(bogus=1), "unknown fields")
    rejects(lambda d: d.update(launch_params="x"), "launch_params must be a list")


# -- the Triton reference ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref",
    ["nocolon", "a b:c", "mod:fn()", "1mod:fn", "mod:1fn", "mod:", ":fn", "mod:fn:extra", "mod..sub:fn", "mod.sub.:fn", "", None, 5, "mod:fn\n"],
)
def test_triton_ref_must_be_module_path_colon_identifier(ref):
    rejects(lambda d: d["triton"].update(ref=ref), "triton ref")


def test_triton_module_and_function_must_match_the_ref():
    rejects(lambda d: d["triton"].update(module="other"), "do not match the ref")
    rejects(lambda d: d["triton"].update(function="other"), "do not match the ref")
    rejects(lambda d: d["triton"].update(extra=1), "unknown fields")
    rejects(lambda d: d.update(triton="my_kernels.sub:fn"), "triton must be an object")
    ok = tampered(lambda d: d["triton"].update(ref="a.b.c:d_1", module="a.b.c", function="d_1"))
    assert CompileSpec.from_dict(ok).triton_module == "a.b.c"


# -- the grid is never evaluated ----------------------------------------------------------------------------------------------------


GOOD_GRIDS = ["(S, 32)", "(S,)", "(cdiv(S, 16), 2*n + 1)", "(S // 2,)", "(-n,)", "(cdiv(S, cdiv(n, 4)), 1, 1)", "((S + 1) * 2,)", "(1,)", "(S, n,)"]


@pytest.mark.parametrize("grid", GOOD_GRIDS)
def test_grids_in_the_restricted_grammar_are_accepted(grid):
    parse_grid(grid)
    assert CompileSpec.from_dict(tampered(lambda d: d.update(grid=grid))).grid == grid


@pytest.mark.parametrize(
    "grid",
    [
        "__import__('os').system('echo pwned')",
        "(S.real,)",
        "(S[0],)",
        "(open('f'),)",
        "(print(1),)",
        "(cdiv(S),)",
        "(cdiv(S, 1, 2),)",
        "(cdiv,)",
        "(S ** 2,)",
        "(S % 2,)",
        "(S / 2,)",
        "(S if n else 1,)",
        "(lambda: 1,)",
        "(S; 1)",
        "(S, n, 1, 1)",
        "()",
        "(S)",  # not a tuple
        "S,",
        "(S,) + (1,)",
        "(S, )) ",
        "(S,",
        "(S,))",
        "(Q,)",  # not a dimension of this spec
        "(1.5,)",
        "(0x10,)",
        "(1e3,)",
        "(S,\n1)",
        "(S,\t1)",
        "(٣,)",  # a non-ASCII digit
        "(S− 1,)",
        "(" + "S+" * 300 + "S,)",
        "(" * 40 + "S" + ")" * 40 + ",",
        "(" + "-" * 40 + "S,)",
        "(" + "9" * 13 + ",)",
        "",
        None,
        5,
        ["(S,)"],
    ],
)
def test_anything_outside_the_grammar_is_rejected_and_never_evaluated(grid):
    rejects(lambda d: d.update(grid=grid))


@pytest.mark.parametrize(
    "grid",
    [
        "(0,)",
        "(-1,)",
        "(4, 0)",
        "(2, 2, -3)",
        "(1 - 1,)",
        "(2 - 3, 4)",
        "(-(2),)",
        "(cdiv(0, 4),)",
        "(0 // 4,)",
        "(0 * 5, 1)",
        "(n - n,)",  # cancels to the static size 0
        "(S, n - n - 1)",
        "(2 * n - 2 * n,)",
        "(cdiv(n - n, 16),)",
        "(" + "0" * 12 + ",)",
    ],
)
def test_a_static_grid_entry_below_one_is_rejected_like_compile_specs_does(grid):
    rejects(lambda d: d.update(grid=grid), "grid sizes must be at least 1")
    with pytest.raises(TilefishSpecError, match="grid sizes must be at least 1"):
        parse_grid(grid)


def test_the_validator_and_compile_specs_agree_on_static_grids():
    def declare(grid):
        @tilefish.extern(triton="m:f", grid=grid)
        def e(x: Tensor[n, bf16]): ...

        return e

    for sizes in [(1,), (0,), (-1,), (4, 0), (8, 1, 1), (2, 2, -3)]:
        text = "(" + ", ".join(map(str, sizes)) + ("," if len(sizes) == 1 else "") + ")"
        if min(sizes) >= 1:
            assert declare(sizes).grid_string == text
            assert CompileSpec.from_dict(tampered(lambda d: d.update(grid=text))).grid == text
        else:
            with pytest.raises(tilefish.TilefishDeclarationError, match="grid sizes must be at least 1"):
                declare(sizes)
            rejects(lambda d: d.update(grid=text), "grid sizes must be at least 1")

    # a bucket that makes an entry static is caught when the spec is generated
    @tilefish.extern(triton="m:f", grid=lambda S: (S - 16,))
    def at_zero(x: Tensor[S, bf16], B: const = S): ...

    with pytest.raises(TilefishSpecError, match="non-positive size 0"):
        compile_specs(at_zero)


@pytest.mark.parametrize("grid", ["(1,)", "(n - n + 1,)", "(2 - 1, 4)", "(S - 1, 1)", "(cdiv(1, 4),)", "(cdiv(n, 16), 1)", "(n * 0 + 3,)"])
def test_static_grid_entries_of_at_least_one_and_symbolic_entries_are_accepted(grid):
    assert CompileSpec.from_dict(tampered(lambda d: d.update(grid=grid))).grid == grid


@pytest.mark.parametrize("grid", ["(S // 0,)", "(cdiv(S, 0),)", "(cdiv(S, -4),)", "(S // -2,)", "(1 // 0, 1)"])
def test_dividing_by_a_non_positive_constant_is_malformed(grid):
    rejects(lambda d: d.update(grid=grid), "the divisor must be positive")


def test_grid_identifiers_must_be_dimensions_of_this_spec():
    rejects(lambda d: d.update(grid="(S, m)"), "not dimensions of this spec")
    rejects(lambda d: d.update(grid="(cdiv(Q, 2),)"), "not dimensions of this spec")
    CompileSpec.from_dict(tampered(lambda d: d.update(grid="(S, n)")))  # both appear in the shapes
    CompileSpec.from_dict(tampered(lambda d: d.update(grid="(S, S)")))


def test_expression_grammar_directly():
    assert parse_expression("2*S + cdiv(n, 16) - 1", "dim") == {"S", "n"}
    assert parse_expression("-(S + 1) // 2", "dim") == {"S"}
    for bad in ("S +", "* S", "S S", "f(S)", "S.x", "cdiv", "cdiv(S)", "S if n", "S ** 2", "1.5", "S,"):
        with pytest.raises(TilefishSpecError):
            parse_expression(bad, "dim")
    assert parse_grid("(a, b, c)") == {"a", "b", "c"}


# -- preconditions and buckets ------------------------------------------------------------------------------------------------------


def test_every_pointer_needs_one_16_byte_alignment_precondition():
    first = next(i for i, p in enumerate(GOOD["preconditions"]) if p["kind"] == "pointer_alignment")
    rejects(lambda d: d["preconditions"].pop(first), "every pointer argument needs exactly one")
    rejects(lambda d: d["preconditions"][first].update(bytes=8), "pointer_alignment must be 16 bytes")
    rejects(lambda d: d["preconditions"].append(copy.deepcopy(d["preconditions"][first])), "pointer_alignment must be 16 bytes")
    rejects(lambda d: d["preconditions"][first].update(arg=d["args"][INDEX["cnt"]]["name"]), "pointer_alignment|which is not an argument")
    rejects(lambda d: d["preconditions"][first].update(index=INDEX["w"]), "which is not an argument")


def test_static_size_preconditions_must_match_the_shapes():
    i = next(i for i, p in enumerate(GOOD["preconditions"]) if p["kind"] == "static_size")
    rejects(lambda d: d["preconditions"][i].update(value=4), "static_size of")
    rejects(lambda d: d["preconditions"][i].update(axis=3), "static_size of")
    rejects(lambda d: d["preconditions"][i].update(arg="x", index=INDEX["x"]), "static_size of")


def test_bucket_preconditions_and_stem_must_match_the_buckets():
    b = next(i for i, p in enumerate(GOOD["preconditions"]) if p["kind"] == "bucket")
    rejects(lambda d: d["preconditions"].pop(b), "every bucket needs exactly one")
    rejects(lambda d: d["preconditions"][b].update(value=32), "disagrees with the spec's buckets")
    rejects(lambda d: d["preconditions"][b].update(buckets=[32, 64]), "disagrees with the spec's buckets")
    rejects(lambda d: d["preconditions"][b].update(dim="T"), "disagrees with the spec's buckets")
    rejects(lambda d: d["preconditions"][b].update(buckets=[32, 16]), "sorted list")
    rejects(lambda d: d["preconditions"].append(copy.deepcopy(d["preconditions"][b])), "disagrees")
    rejects(lambda d: d["buckets"].update(S=32), "stem|disagrees")
    rejects(lambda d: d.update(stem="kernel.S99"), "does not match op")
    rejects(lambda d: d.update(stem="kernel; rm"), "stem must look like")
    rejects(lambda d: d.update(op="other"), "does not match op")
    rejects(lambda d: d["buckets"].update({"1bad": 3}), "identifier")
    rejects(lambda d: d["buckets"].update(S=0), "must be an int")
    rejects(lambda d: d["buckets"].update(S=True), "must be an int")


def test_unknown_precondition_kinds_and_fields_are_rejected():
    rejects(lambda d: d["preconditions"].append({"kind": "run_code", "code": "x"}), "precondition kind must be one of")
    rejects(lambda d: d["preconditions"][0].update(evil=1), "unknown fields")
    rejects(lambda d: d["preconditions"][0].pop("kind"), "precondition kind must be")
    rejects(lambda d: d["preconditions"][0].pop("bytes"), "is missing")
    rejects(lambda d: d["preconditions"][0].update(index="0"), "must be an int")
    rejects(lambda d: d.update(preconditions={}), "preconditions must be a list")


# -- bare exceptions never escape ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("junk", [None, 5, "text", [1, 2, 3], [], (), {}, {"schema": 1}, 3.5, True])
def test_non_object_and_incomplete_input_is_a_spec_error(junk):
    with pytest.raises(TilefishSpecError):
        CompileSpec.from_dict(junk)


@pytest.mark.parametrize("key", [k for k in GOOD if k != "dims"])  # the table of dimensions is optional: an older spec has none
def test_every_missing_top_level_key_is_a_spec_error(key):
    if key == "schema":
        rejects(lambda d: d.pop("schema"), "unsupported compile spec schema")
        return
    rejects(lambda d: d.pop(key), "is missing")


@pytest.mark.parametrize("key", [k for k in GOOD if k != "schema"])
@pytest.mark.parametrize("junk", [None, 5, "text", [1], {"a": 1}, 3.5])
def test_wrongly_typed_values_are_spec_errors_not_bare_exceptions(key, junk):
    d = copy.deepcopy(GOOD)
    if key == "num_stages" and junk == 5:
        return  # a valid value
    d[key] = junk
    with pytest.raises(TilefishSpecError):
        CompileSpec.from_dict(d)


def test_unknown_top_level_keys_are_rejected():
    rejects(lambda d: d.update(evil="x"), "unknown fields")


def test_nested_from_dicts_are_strict_too():
    for bad in ({}, {"kind": "bucket"}, {"kind": "pointer_alignment", "arg": "x", "index": "0", "bytes": 16}, None, 3):
        with pytest.raises(TilefishSpecError):
            Precondition.from_dict(bad)
    for bad in ({}, {"index": 0, "name": "x", "kind": "pointer", "type": "*bf16"}, {"index": -1, "name": "x", "kind": "constexpr", "type": "constexpr", "value": 1}, None):
        with pytest.raises(TilefishSpecError):
            ArgSpec.from_dict(bad)
    for bad in ({}, {"name": "x"}, {"name": "x", "type": "u64", "value": "0"}, {"name": "1", "type": "u64"}, None):
        with pytest.raises(TilefishSpecError):
            LaunchParam.from_dict(bad)
    assert Precondition.from_dict({"kind": "pointer_alignment", "arg": "x", "index": 0, "bytes": 16}).bytes == 16
    assert LaunchParam.from_dict({"name": "global_scratch", "type": "u64", "value": 0}).value == 0
    assert ArgSpec.from_dict({"index": 0, "name": "B", "kind": "constexpr", "type": "constexpr", "value": {"float": "-inf"}}).value == float("-inf")


def test_from_json_rejects_bad_json_and_non_standard_constants():
    for text in ("{not json", "", "[]", '"x"', "null", "5", '{"schema": 1, "x": NaN}', '{"schema": 1, "x": Infinity}', '{"schema": 1, "x": -Infinity}'):
        with pytest.raises(TilefishSpecError):
            CompileSpec.from_json(text)
    with pytest.raises(TilefishSpecError, match="JSON constant NaN is not allowed"):
        CompileSpec.from_json('{"schema": 1, "x": NaN}')
    deep = "[" * 100000 + "]" * 100000
    with pytest.raises(TilefishSpecError):
        CompileSpec.from_json(deep)
    assert CompileSpec.from_json(SPEC.to_json()) == SPEC
    doc = json.loads(SPEC.to_json())
    assert doc["constants"][str(INDEX["E"])] == {"float": "inf"}
