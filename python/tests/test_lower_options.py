"""What the lowering decides per binary and per kernel: the warps of each bucket, the FMA contraction, and which casts exist.

* Each bucket of a bucketed ``D`` gets the warp count of its own ``BLOCK_D`` (one warp per 256 lanes, 1 to 16), so the
  bits of the ``D = 96`` binary do not depend on which other buckets a kernel declares. ``num_warps=N`` pins every bucket.
* A generated kernel writes ``enable_fp_fusion`` into its specs (``True``, Triton's default, or ``False`` for
  ``@tilefish.kernel(fp_fusion=False)``); an ``extern`` has no such key.
* A float to integer cast is rejected (rule ``cast.float-to-int``); float to ``e4m3``/``e5m2`` and the rest stay.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import copy
import dataclasses
import importlib
import json

import pytest
from built_fakes import scale
from lower_cases import T, layernorm, softmax

import tilefish
from tilefish import Dim, Tensor, TilefishDeclarationError, TilefishLoweringError, TilefishSpecError, bf16, e4m3, e5m2, f16, f32, i32, i64, u8
from tilefish.aot import CompileSpec, compile_specs, declaration_digest
from tilefish.decorators import ExternOp


def rows_kernel(d_dim, **options):
    rows = Dim("R")

    @tilefish.kernel(**options)
    def rowwise(x: Tensor[rows, d_dim, f32]) -> Tensor[rows, d_dim, f32]:
        return x - tilefish.mean(x, axis=-1, keepdims=True)

    return rowwise


# -- warps per bucket ----------------------------------------------------------------------------------------------------------------


def test_each_bucket_gets_the_warps_of_its_own_block():
    low = rows_kernel(Dim("D", buckets=[16, 96, 256, 257, 1000, 4096, 65536])).lower()
    # BLOCK_D = 16, 128, 256, 512, 1024, 4096, 65536 lanes: one warp per 256, at least 1, at most 16
    assert [s.num_warps for s in low.specs()] == [1, 1, 1, 2, 4, 16, 16]
    assert [s.block for s in low.specs()] == [(32, 1, 1), (32, 1, 1), (32, 1, 1), (64, 1, 1), (128, 1, 1), (512, 1, 1), (512, 1, 1)]
    assert low.bucket_num_warps == {".D16": 1, ".D96": 1, ".D256": 1, ".D257": 2, ".D1000": 4, ".D4096": 16, ".D65536": 16}
    assert low.num_warps == 16  # the largest bucket's, for a reader of the lowered kernel; no binary falls back to it


def test_the_warps_of_a_bucket_do_not_depend_on_the_other_buckets():
    small = rows_kernel(Dim("D", buckets=[96, 512])).lower()
    large = rows_kernel(Dim("D", buckets=[96, 8192])).lower()
    alone = rows_kernel(Dim("D", 96)).lower()
    assert small.specs()[0].num_warps == large.specs()[0].num_warps == alone.specs()[0].num_warps == 1
    assert [s.num_warps for s in small.specs()] == [1, 2] and [s.num_warps for s in large.specs()] == [1, 16]
    # the specs of the D = 96 binary are the same apart from what names the other buckets
    a, b = small.specs()[0].to_dict(), large.specs()[0].to_dict()
    assert a["num_warps"] == b["num_warps"] and a["block"] == b["block"] and a["constants"] == b["constants"]


def test_an_expression_of_a_bucketed_dimension_and_two_bucketed_dimensions():
    F = Dim("F", buckets=[64, 1024])
    G = Dim("G", buckets=[32, 512])
    low = rows_kernel(F + G).lower()
    assert [(s.stem, s.num_warps) for s in low.specs()] == [
        ("rowwise.F64.G32", 1), ("rowwise.F64.G512", 4), ("rowwise.F1024.G32", 8), ("rowwise.F1024.G512", 8)
    ]  # D = F + G = 96, 576, 1056, 1536: BLOCK_D = 128, 1024, 2048, 2048, in the order of the specs


def test_a_static_d_and_an_elementwise_kernel_have_one_value_and_no_table():
    assert softmax.lower().bucket_num_warps == {} and softmax.lower().num_warps == 4
    low = tilefish.lower(layernorm)
    assert low.bucket_num_warps == {} and {s.num_warps for s in low.specs()} == {4}

    @tilefish.kernel
    def flat(x: Tensor[T, f32]) -> Tensor[T, f32]:
        return x * 2.0

    assert flat.lower().bucket_num_warps == {} and flat.lower().num_warps == 4


def test_an_explicit_num_warps_pins_every_bucket():
    k = rows_kernel(Dim("D", buckets=[96, 512, 4096]), num_warps=2)
    low = k.lower()
    assert {s.num_warps for s in low.specs()} == {2} and low.bucket_num_warps == {} and low.num_warps == 2
    assert {s.num_warps for s in k.lower(num_warps=8).specs()} == {8}
    assert {s.num_warps for s in rows_kernel(Dim("D", buckets=[96, 4096])).lower(num_warps=1).specs()} == {1}
    assert "num_warps: 2\n" in low.source


def test_the_source_says_the_warps_of_each_bucket():
    low = rows_kernel(Dim("D", buckets=[96, 4096])).lower()
    assert "# num_warps: 1 for D96, 16 for D4096\n" in low.source
    assert "num_warps: 4\n" in softmax.lower().source  # unchanged for a kernel with one value


def test_per_bucket_warps_are_part_of_the_declaration_and_verified():
    low = rows_kernel(Dim("D", buckets=[96, 4096])).lower()
    pinned = rows_kernel(Dim("D", buckets=[96, 4096]), num_warps=16).lower()
    assert declaration_digest(low.extern) != declaration_digest(pinned.extern)
    for spec in low.specs():
        back = CompileSpec.from_json(spec.to_json())
        back.verify_against(low.extern)
        assert back.verified
    edited = CompileSpec.from_dict({**low.specs()[0].to_dict(), "num_warps": 16, "block": [512, 1, 1]})
    with pytest.raises(TilefishSpecError, match="first difference at num_warps"):
        edited.verify_against(low.extern)  # a spec of the first bucket with the warps of the second


def test_the_per_bucket_table_is_checked_where_an_extern_is_built():
    low = rows_kernel(Dim("D", buckets=[96, 4096])).lower()
    ext = low.extern

    def with_table(table):
        return ExternOp(ext.fn, ext.name, ext.signature, ext.triton, ext.num_warps, ext.num_stages, ext.grid, bucket_num_warps=table)

    with pytest.raises(TilefishSpecError, match=r"bucket_num_warps names buckets that the op does not have: \['.D100'\]"):
        compile_specs(with_table({".D96": 1, ".D100": 2}))
    for table, message in (
        ({"D96": 1}, "stem suffixes such as '.D96'"),
        ({".D96": 3}, "power of two from 1 to 32"),
        ({".D96": True}, "power of two from 1 to 32"),
        ({".D96": 64}, "power of two from 1 to 32"),
        ({5: 1}, "stem suffixes"),
    ):
        with pytest.raises(TilefishDeclarationError, match=message):
            with_table(table)
    assert [s.num_warps for s in compile_specs(with_table({".D96": 2}))] == [2, ext.num_warps]  # a bucket without an entry has num_warps


def test_an_extern_has_no_table_and_its_specs_are_what_they_were():
    (spec,) = compile_specs(scale)
    assert scale.bucket_num_warps == {} and scale.fp_fusion is None
    assert spec.num_warps == 4 and spec.enable_fp_fusion is None
    assert "enable_fp_fusion" not in spec.to_dict() and spec.ast_source_inputs().options == {"num_warps": 4, "num_stages": 3}


# -- the contraction ---------------------------------------------------------------------------------------------------------------------


def test_a_generated_kernel_writes_triton_default_into_its_spec():
    (spec,) = layernorm.lower().specs()
    assert spec.enable_fp_fusion is True
    assert spec.to_dict()["enable_fp_fusion"] is True
    assert spec.ast_source_inputs().options == {"num_warps": 4, "num_stages": 3, "enable_fp_fusion": True}
    assert layernorm.lower().fp_fusion is True and layernorm.fp_fusion is True


def test_fp_fusion_false_is_the_same_source_and_another_spec():
    on, off = layernorm.lower(), layernorm.lower(fp_fusion=False)
    (spec_on,), (spec_off,) = on.specs(), off.specs()
    assert off.fp_fusion is False and spec_off.enable_fp_fusion is False
    assert spec_off.ast_source_inputs().options["enable_fp_fusion"] is False
    assert off.source.replace("# fp_fusion: False (no FMA contraction: a * b + c rounds twice)\n", "") == on.source  # the code is the same
    assert "# fp_fusion: False" in off.source and "fp_fusion" not in on.source  # and a reviewer is told
    assert declaration_digest(on.extern) != declaration_digest(off.extern)
    assert spec_on._fingerprint() != spec_off._fingerprint()


def test_the_kernel_decorator_and_lower_take_fp_fusion():
    @tilefish.kernel(fp_fusion=False)
    def strict(x: Tensor[T, f32], y: Tensor[T, f32], z: Tensor[T, f32]) -> Tensor[T, f32]:
        return x * y + z

    assert strict.fp_fusion is False
    assert strict.lower().fp_fusion is False
    assert strict.lower(fp_fusion=True).fp_fusion is True  # lower() overrides the declaration
    assert tilefish.lower(strict.lower().kernel).fp_fusion is False
    assert [s.enable_fp_fusion for s in strict.lower().specs()] == [False]
    with pytest.raises(TilefishDeclarationError, match="fp_fusion must be True or False"):
        tilefish.kernel(fp_fusion=0)
    with pytest.raises(TilefishDeclarationError, match="fp_fusion must be True or False"):
        tilefish.kernel(fp_fusion=None)
    with pytest.raises(TilefishLoweringError, match="fp_fusion must be True or False") as e:
        strict.lower(fp_fusion="no")
    assert e.value.rule == "fp-fusion"


def test_the_contraction_is_in_the_json_and_in_verification():
    low = layernorm.lower()
    (spec,) = low.specs()
    doc = json.loads(spec.to_json())
    assert doc["enable_fp_fusion"] is True
    back = CompileSpec.from_json(spec.to_json())
    assert back.enable_fp_fusion is True and back == CompileSpec.from_dict(spec.to_dict())
    back.verify_against(low.extern)
    # a stored spec whose contraction was flipped is a valid spec, but not this declaration's
    flipped = copy.deepcopy(doc)
    flipped["enable_fp_fusion"] = False
    edited = CompileSpec.from_dict(flipped)
    with pytest.raises(TilefishSpecError, match="first difference at enable_fp_fusion: the spec has False, the declaration gives True"):
        edited.verify_against(low.extern)
    assert not edited.verified
    # a spec without the key (written before it existed) is not this declaration's either
    removed = copy.deepcopy(doc)
    del removed["enable_fp_fusion"]
    with pytest.raises(TilefishSpecError, match="first difference at enable_fp_fusion"):
        CompileSpec.from_dict(removed).verify_against(low.extern)


def test_enable_fp_fusion_in_a_serialized_spec_is_validated():
    (spec,) = layernorm.lower().specs()
    doc = json.loads(spec.to_json())
    for bad in (1, 0, "true", None, [True]):
        d = copy.deepcopy(doc)
        d["enable_fp_fusion"] = bad
        with pytest.raises(TilefishSpecError, match="enable_fp_fusion must be true or false"):
            CompileSpec.from_dict(d)
    legacy = copy.deepcopy(doc)  # schema 1 has no such key
    legacy["schema"] = 1
    del legacy["declaration_digest"]
    with pytest.raises(TilefishSpecError, match="unknown fields .*enable_fp_fusion"):
        CompileSpec.from_dict(legacy)


def test_an_edited_spec_loses_its_verification_when_the_contraction_changes():
    (spec,) = layernorm.lower().specs()
    assert spec.verified
    changed = dataclasses.replace(spec, enable_fp_fusion=False)
    assert not changed.verified
    with pytest.raises(TilefishSpecError, match="not been verified"):
        changed.ast_source_inputs()


def test_an_extern_op_validates_its_contraction():
    low = layernorm.lower()
    ext = low.extern

    def make(value):
        return ExternOp(ext.fn, ext.name, ext.signature, ext.triton, ext.num_warps, ext.num_stages, ext.grid, fp_fusion=value)

    assert make(None).fp_fusion is None and make(False).fp_fusion is False
    with pytest.raises(TilefishDeclarationError, match="fp_fusion must be True, False or None"):
        make(1)
    assert "enable_fp_fusion" not in compile_specs(make(None))[0].to_dict()


# -- casts ----------------------------------------------------------------------------------------------------------------------------------


def cast_kernel(src, dst):
    @tilefish.kernel
    def cast(x: Tensor[T, src]) -> Tensor[T, dst]:
        return x.to(dst)

    return cast


@pytest.mark.parametrize("src", [f32, bf16, f16], ids=str)
@pytest.mark.parametrize("dst", [u8, i32, i64], ids=str)
def test_a_float_to_integer_cast_is_not_lowered(src, dst):
    with pytest.raises(TilefishLoweringError) as e:
        cast_kernel(src, dst).lower()
    assert e.value.rule == "cast.float-to-int"
    message = str(e.value)
    assert f"casting {src} to {dst} is not lowered" in message
    assert "rounding and saturation modes of a float to integer cast are not supported yet" in message
    assert "numerics contract" in message


def test_the_rule_applies_to_every_template():
    for template in ("elementwise", None):
        with pytest.raises(TilefishLoweringError) as e:
            cast_kernel(f32, i32).lower(template=template)
        assert e.value.rule == "cast.float-to-int"
    rows = Dim("R")

    @tilefish.kernel
    def rows_to_int(x: Tensor[rows, Dim("D", 64), f32]) -> Tensor[rows, Dim("D", 64), i32]:
        return x.to(i32)

    with pytest.raises(TilefishLoweringError) as e:
        rows_to_int.lower()
    assert e.value.rule == "cast.float-to-int"


@pytest.mark.parametrize(
    "src, dst",
    [(f32, bf16), (f32, f16), (bf16, f32), (f16, f32), (f32, e4m3), (f32, e5m2), (bf16, e4m3), (i32, f32), (i64, f32), (u8, f32), (i64, i32), (i32, u8), (u8, i32), (i32, i64)],
    ids=lambda d: str(d),
)
def test_the_other_casts_are_unchanged(src, dst):
    low = cast_kernel(src, dst).lower()
    assert ".to(tl." in low.source


def test_a_float_to_fp8_cast_stays_and_is_documented_as_satfinite():
    low = cast_kernel(f32, e4m3).lower()
    assert "v1 = v0.to(tl.float8e4nv)  # rounding point" in low.source
    assert "satfinite" in importlib.import_module("tilefish.lower").__doc__  # the numerics contract says what the cast does


# -- the source of a generated kernel -----------------------------------------------------------------------------------------------------


def test_the_source_sha256_of_a_generated_kernel_round_trips_and_is_verified():
    import hashlib

    low = layernorm.lower()
    (spec,) = low.specs()
    doc = json.loads(spec.to_json())
    assert doc["source_sha256"] == spec.source_sha256 == low.source_sha256 == hashlib.sha256(low.source.encode("utf-8")).hexdigest()
    back = CompileSpec.from_json(spec.to_json())
    assert back.source_sha256 == low.source_sha256 and back == spec and back.to_dict() == spec.to_dict()
    back.verify_against(low.extern)
    other = copy.deepcopy(doc)
    other["source_sha256"] = "0" * 64  # a valid spec, of other source: not this declaration's
    with pytest.raises(TilefishSpecError, match="first difference at source_sha256"):
        CompileSpec.from_dict(other).verify_against(low.extern)
    removed = copy.deepcopy(doc)
    del removed["source_sha256"]
    assert CompileSpec.from_dict(removed).source_sha256 is None  # (optional in a document)
    with pytest.raises(TilefishSpecError, match="first difference at source_sha256"):
        CompileSpec.from_dict(removed).verify_against(low.extern)


def test_an_extern_has_no_source_sha256_and_a_generated_kernel_always_has_one():
    assert "source_sha256" not in compile_specs(scale)[0].to_dict() and compile_specs(scale)[0].source_sha256 is None
    for k in (layernorm, softmax):
        assert len(k.lower().specs()[0].source_sha256) == 64


def test_the_source_sha256_is_in_the_declaration_digest_and_the_fingerprint():
    ext = layernorm.lower().extern

    def make(value):
        return ExternOp(ext.fn, ext.name, ext.signature, ext.triton, ext.num_warps, ext.num_stages, ext.grid, source_sha256=value)

    a, b = "a" * 64, "b" * 64
    assert len({declaration_digest(make(a)), declaration_digest(make(b)), declaration_digest(make(None))}) == 3
    sa, sb = compile_specs(make(a))[0], compile_specs(make(b))[0]
    assert sa.declaration_digest != sb.declaration_digest and sa._fingerprint() != sb._fingerprint()
    assert sa.source_sha256 == a and sa.to_dict()["source_sha256"] == a
    stale = dataclasses.replace(sa, source_sha256=b)  # an edit of a verified spec is not verified
    assert not stale.verified


def test_the_source_sha256_of_an_extern_must_be_a_sha256():
    ext = layernorm.lower().extern

    def make(value):
        return ExternOp(ext.fn, ext.name, ext.signature, ext.triton, ext.num_warps, ext.num_stages, ext.grid, source_sha256=value)

    assert make("0123456789abcdef" * 4).source_sha256 == "0123456789abcdef" * 4
    for bad in ("", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "a" * 63 + "\n", "a" * 64 + "\n", " " + "a" * 63, 5, b"a" * 64, ["a" * 64]):
        with pytest.raises(TilefishDeclarationError, match="source_sha256 must be 64 lowercase hex digits"):
            make(bad)


def test_the_source_sha256_of_a_serialized_spec_must_be_a_sha256():
    (spec,) = layernorm.lower().specs()
    doc = json.loads(spec.to_json())
    for bad in ("", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "a" * 64 + "\n", 5, None, ["a" * 64]):
        d = copy.deepcopy(doc)
        d["source_sha256"] = bad
        with pytest.raises(TilefishSpecError, match="source_sha256 must be 64 lowercase hex digits"):
            CompileSpec.from_dict(d)
    legacy = copy.deepcopy(doc)  # schema 1 has no such key
    legacy["schema"] = 1
    del legacy["declaration_digest"]
    with pytest.raises(TilefishSpecError, match="unknown fields .*source_sha256"):
        CompileSpec.from_dict(legacy)
