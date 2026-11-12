"""A typed constexpr compiles in a value of its declared dtype: const[f32] is a float, an integer dtype an int."""

import copy

import pytest

import tilefish
from tilefish import Dim, Tensor, TilefishSpecError, bf16, const, f32, i32
from tilefish.aot import CompileSpec, _typed_const, compile_specs, declaration_digest
from tilefish.signature import Param

n = Dim("n", divisible_by=16)


def test_a_typed_const_takes_the_type_of_its_dtype_even_when_it_comes_from_a_dimension():
    Sb = Dim("Sb", buckets=[16, 32])
    Dn = Dim("Dn", 64)

    @tilefish.extern(triton="m:f", grid=(1,))
    def typed(
        x: Tensor[Sb, bf16],
        *,
        F: const[f32] = Sb,  # a bucketed int dim
        G: const[f32] = Dn,  # a static int dim
        H: const[f32] = 3,  # an int literal
        I: const[i32] = Sb,
        J: const[tilefish.i64] = tilefish.cdiv(Sb, 4),
        U: const = Sb,  # untyped: unchanged
        V: const = 2.5,
    ): ...

    specs = compile_specs(typed)
    assert [s.stem for s in specs] == ["typed.Sb16", "typed.Sb32"]
    for spec, bucket in zip(specs, (16, 32)):
        values = {a.name: a.value for a in spec.args if a.kind == "constexpr"}
        assert values == {"F": float(bucket), "G": 64.0, "H": 3.0, "I": bucket, "J": bucket // 4, "U": bucket, "V": 2.5}
        kinds = {k: type(v) for k, v in values.items()}
        assert kinds == {"F": float, "G": float, "H": float, "I": int, "J": int, "U": int, "V": float}
        # the constants dict, the args and the JSON agree
        assert {a.name: spec.constants[(a.index,)] for a in spec.args if a.kind == "constexpr"} == values
        back = CompileSpec.from_json(spec.to_json())
        assert back == spec and back.verify_against(typed) is None
        assert [type(a.value) for a in back.args[1:]] == [float, float, float, int, int, int, float]


def test_the_untyped_int_a_float_const_used_to_get_is_caught_by_verify_against():
    Sb = Dim("Sb2", buckets=[16, 32])

    @tilefish.extern(triton="m:f", grid=(1,))
    def typed(x: Tensor[Sb, bf16], *, F: const[f32] = Sb): ...

    spec16, _ = compile_specs(typed)
    index = {a.name: a.index for a in spec16.args}
    doc = copy.deepcopy(spec16.to_dict())
    doc["constants"][str(index["F"])] = 16  # an int, as before
    doc["args"][index["F"]]["value"] = 16
    old_style = CompileSpec.from_dict(doc)  # it validates ...
    with pytest.raises(TilefishSpecError, match=rf"first difference at args\[{index['F']}\]\.value: the spec has 16, the declaration gives 16\.0"):
        old_style.verify_against(typed)  # ... but it is not what the declaration generates
    assert spec16.verify_against(typed) is None


def param(dtype):
    return Param("c", 0, "const", const_dtype=dtype, default=0)


def test_an_integer_dtype_emits_an_int_and_rejects_a_non_integral_value():
    assert _typed_const("op", param(i32), 4) == 4 and type(_typed_const("op", param(i32), 4)) is int
    assert type(_typed_const("op", param(i32), 4.0)) is int  # integral: fine
    assert _typed_const("op", param(tilefish.u8), 255.0) == 255
    for bad in (2.5, float("inf"), float("nan"), -0.5):
        with pytest.raises(TilefishSpecError, match=r"constexpr 'c' is const\[i32\] but its value .* is not an integer"):
            _typed_const("op", param(i32), bad)
    with pytest.raises(TilefishSpecError, match="does not fit"):
        _typed_const("op", param(f32), 10**400)


def test_float_and_untyped_consts():
    assert _typed_const("op", param(f32), 3) == 3.0 and type(_typed_const("op", param(f32), 3)) is float
    assert _typed_const("op", param(None), 3) == 3 and type(_typed_const("op", param(None), 3)) is int  # untyped
    assert _typed_const("op", param(None), 2.5) == 2.5
    assert _typed_const("op", param(f32), True) is True  # a bool is left alone (a typed const cannot declare one)


def test_the_digest_hashes_the_typed_value_of_a_scalar_default():
    def digest(ann, value):
        @tilefish.extern(triton="m:f", grid=(1,))
        def e(x: Tensor[n, bf16], *, C: ann = value): ...

        return declaration_digest(e)

    # declared as an int but typed f32, the default is already 3.0 at declaration: the same declaration
    assert digest(const[f32], 3) == digest(const[f32], 3.0)
    assert digest(const[i32], 3) != digest(const[f32], 3)
    assert digest(const, 3) != digest(const[i32], 3)
    # a dimension default is hashed with its dtype: S as f32, as i32 and untyped are different declarations
    Sd = Dim("Sd", buckets=[16, 32])
    digests = {digest(const[f32], Sd), digest(const[i32], Sd), digest(const, Sd)}
    assert len(digests) == 3
