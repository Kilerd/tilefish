"""The type of a stride scalar is the declaration's: ``StrideOf(..., dtype=i32 | i64)``, independent of ``index_bits``.

``index_bits`` says how wide the offsets are that the kernel computes; the width of a stride *parameter* is another fact. A kernel that takes
``i32`` strides and widens them before it multiplies (``row.to(tl.int64) * s_row``) computes 64-bit offsets from 32-bit strides: it is
``index_bits=64`` (no ``max_elements`` / ``max_extent`` limit) with ``StrideOf("x", 0, dtype=i32)``. Without ``dtype=`` a stride is as wide as
the offsets, as before (``i64`` under ``index_bits=64``, else ``i32``), and a ``dtype=`` that says that default declares the same thing: the
same spec and the same declaration digest. The type is the spec's ``signature`` and argument type, the manifest's ``abi`` type, the PTX
parameter a prebuilt binary must have, and the range a launch checks. No GPU here.
"""

import json

import pytest
from built_fakes import fake_build
from fake_torch import FakeTensor, prepare, verdict

import tilefish
from tilefish import Dim, Manifest, Out, StrideOf, Strided, Tensor, TilefishBuildError, TilefishDeclarationError, bf16, f32, i32, i64
from tilefish._binary import check_ptx_abi
from tilefish.aot import compile_specs, declaration_digest, write_manifest
from tilefish.signature import parse_signature

R = Dim("R")


def strided(dtype=None, index_bits=64, ref="sd_kernels:copy"):
    @tilefish.extern(triton=ref, grid=lambda R: (R,), index_bits=index_bits, name="strided_rows")
    def op(x: Strided[Tensor[R, 64, bf16]], y: Out[Tensor[R, 64, bf16]], x_s0: StrideOf("x", 0, dtype=dtype)): ...

    return op


def stride_type(op):
    (spec,) = compile_specs(op)
    (arg,) = [a for a in spec.args if a.name == "x_s0"]
    assert spec.signature["x_s0"] == arg.type == arg.dtype
    return arg.type


def limits(op):
    (spec,) = compile_specs(op)
    return sorted(p.kind for p in spec.preconditions if p.kind in ("max_elements", "max_extent"))


def test_an_i32_stride_of_a_kernel_that_indexes_in_i64():
    default, narrow = strided(), strided(dtype=i32)
    assert stride_type(default) == "i64" and stride_type(narrow) == "i32"
    assert limits(narrow) == limits(default) == []  # still index_bits=64: no limit on the offsets
    assert declaration_digest(narrow) != declaration_digest(default)
    for op in (default, narrow):
        (spec,) = compile_specs(op)
        spec.verify_against(op)
    (spec,) = compile_specs(narrow)
    with pytest.raises(tilefish.TilefishSpecError):
        spec.verify_against(default)


def test_a_dtype_that_says_the_default_declares_the_same_thing():
    for index_bits, same in ((64, i64), (32, i32)):
        plain, explicit = strided(index_bits=index_bits), strided(dtype=same, index_bits=index_bits)
        assert declaration_digest(plain) == declaration_digest(explicit)
        assert [s.to_dict() for s in compile_specs(plain)] == [s.to_dict() for s in compile_specs(explicit)]


def test_an_i64_stride_of_a_kernel_that_indexes_in_i32_keeps_its_limit():
    op = strided(dtype=i64, index_bits=32)
    assert stride_type(op) == "i64" and limits(op) == ["max_elements", "max_extent"]  # the offsets are still i32: bounded


def test_a_shared_stride_takes_a_dtype():
    @tilefish.extern(triton="sd_kernels:pair", grid=lambda R: (R,), index_bits=64)
    def pair(
        a: Strided[Tensor[R, 64, bf16]], b: Strided[Tensor[R, 64, bf16]], y: Out[Tensor[R, 64, bf16]], s: StrideOf(("a", "b"), 0, dtype=i32)
    ): ...

    (spec,) = compile_specs(pair)
    (arg,) = [a for a in spec.args if a.name == "s"]
    assert arg.type == "i32" and arg.stride_of == (("a", 0), ("b", 0))


@pytest.mark.parametrize("bad", [f32, bf16, tilefish.u8, "i32", 32, True])
def test_the_dtype_is_i32_or_i64(bad):
    with pytest.raises(TilefishDeclarationError, match="StrideOf: dtype is the type of the stride scalar, tilefish.i32 or tilefish.i64"):
        StrideOf("x", 0, dtype=bad)


def test_the_annotation_shows_its_dtype():
    assert repr(StrideOf("x", 0, dtype=i32)) == "StrideOf('x', 0, dtype=i32)"
    assert repr(StrideOf(("a", "b"), 1, divisible_by=8, dtype=i64)) == "StrideOf(('a', 'b'), 1, divisible_by=8, dtype=i64)"

    def stub(x: Strided[Tensor[R, 64, bf16]], x_s0: StrideOf("x", 0, dtype=i32)): ...

    sig = parse_signature(stub, what="extern op", name="stub", allow_strided=True)
    assert str(sig.params[1]) == "x_s0: StrideOf('x', 0, dtype=i32)"


def test_the_manifest_records_the_type(tmp_path):
    (spec,) = compile_specs(strided(dtype=i32))
    write_manifest([fake_build(spec)], tmp_path / "manifest.json", cubins=True)
    entry = json.loads((tmp_path / "manifest.json").read_text())["kernels"][0]
    (param,) = [p for p in entry["abi"]["params"] if p["name"] == "x_s0"]
    assert (param["type"], param["ptx_type"], param["size"]) == ("i32", "u32", 4) and param["stride_of"] == {"arg": "x", "axis": 0}
    Manifest.load(tmp_path).built_kernel(spec, "sm_90", pinned=True)  # and a replay takes the entry as the spec's


def test_a_launch_checks_the_range_of_the_declared_type():
    """The stride of a view of ``2**31`` elements per row: an ``i32`` stride cannot carry it, an ``i64`` one can (neither has an offset limit)."""
    wide_row = 2**31
    view = FakeTensor((2, 64), "bf16", 1 << 44, strides=(wide_row, 1))
    prepared, error = prepare(strided(dtype=i32), [view], next_slab=1 << 50)
    assert prepared is None and error.kind == "shape" and error.arg == "x_s0" and "32-bit signed integer" in str(error)
    prepared, error = prepare(strided(), [view], next_slab=1 << 50)
    assert error is None and verdict(prepared, error)["params"][2] == wide_row
    prepared, error = prepare(strided(dtype=i32), [FakeTensor((2, 64), "bf16", 1 << 44, strides=(2**31 - 1, 1))], next_slab=1 << 50)
    assert error is None and prepared.scalars["x_s0"] == 2**31 - 1


PTX = """.version 8.8
.target sm_90
.address_size 64

.visible .entry strided_rows(
\t.param .u64 x,
\t.param .u64 y,
\t.param .u32 x_s0
)
{
\tret;
}
"""


def test_a_prebuilt_kernel_with_i32_strides_and_i64_offsets_is_declared_as_it_is(tmp_path):
    """The case that made the type a fact of its own: a prebuilt kernel whose stride parameters are ``.u32`` and whose offsets are 64-bit."""
    path = tmp_path / "strided_rows.ptx"
    path.write_text(PTX)

    def declare(dtype):
        @tilefish.extern(ptx=path, symbol="strided_rows", block=(64,), grid=lambda R: (R,), index_bits=64)
        def strided_rows(x: Strided[Tensor[R, 64, bf16]], y: Out[Tensor[R, 64, bf16]], x_s0: StrideOf("x", 0, dtype=dtype)): ...

        (spec,) = compile_specs(strided_rows)
        return spec

    check_ptx_abi(declare(i32), PTX, "strided_rows")
    with pytest.raises(TilefishBuildError, match=r"does not have the C ABI its stub declares: it takes \['u64', 'u64', 'u32'\]"):
        check_ptx_abi(declare(None), PTX, "strided_rows")  # the default under index_bits=64 is an i64: not what the PTX takes
