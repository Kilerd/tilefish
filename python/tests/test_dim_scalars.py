"""A dimension scalar may be 64 bits wide: ``n_elem: i64[n]`` (an extern only).

A parameter annotated with a dimension is an ``i32``, as it always was. ``i64[n]`` is the same scalar passed as an ``i64``: its value is what the
tensors of the call bind the dimension to (derived from the shapes, never given by the call, unlike a fixed argument per call), it is range-checked
as an ``i64`` at a launch, and it is a ``scalar`` of type ``i64`` in the spec and the manifest. ``i32[n]`` is ``n``: the same spec and the same
declaration digest. No GPU here.
"""

import json

import pytest
from built_fakes import fake_build
from fake_torch import FakeTensor, prepare, verdict

import tilefish
from tilefish import Dim, Manifest, Out, Tensor, TilefishBuildError, TilefishDeclarationError, bf16, const, f32, i32, i64, u8
from tilefish._binary import check_ptx_abi
from tilefish.aot import CompileSpec, compile_specs, declaration_digest, write_manifest
from tilefish.signature import parse_signature
from tilefish.types import DimScalarType

n = Dim("n")
n16 = Dim("n16", divisible_by=16)
R = Dim("R")
C = Dim("C")


def wide(dtype=i64, index_bits=64, dim=n):
    @tilefish.extern(triton="ds_kernels:add_one", grid=lambda n_elem: (tilefish.cdiv(n_elem, 1024),), index_bits=index_bits, name="add_one")
    def op(x: Tensor[dim, u8], y: Out[Tensor[dim, u8]], n_elem: dtype[dim], BLOCK: const = 1024): ...

    return op


def narrow():
    @tilefish.extern(triton="ds_kernels:add_one", grid=lambda n: (tilefish.cdiv(n, 1024),), index_bits=64, name="add_one")
    def op(x: Tensor[n, u8], y: Out[Tensor[n, u8]], n_elem: n, BLOCK: const = 1024): ...

    return op


def scalar_arg(op, name="n_elem"):
    (spec,) = compile_specs(op)
    (arg,) = [a for a in spec.args if a.name == name]
    return spec, arg


def test_the_annotation():
    assert i64[n] == DimScalarType(n, i64) and repr(i64[n]) == "i64[n]" and repr(i64[2 * R + 1]) == "i64[2*R + 1]"
    assert i32[n] is n  # an i32 dimension scalar is the dimension itself

    def stub(x: Tensor[R, C, bf16], k: i64[R * 0 + C], m: R): ...

    sig = parse_signature(stub, what="extern op", name="stub", allow_wide_dims=True)
    assert [str(p) for p in sig.params[1:]] == ["k: i64[C]", "m: R"]
    assert sig.params[1].dim_dtype == i64 and sig.params[2].dim_dtype is None


@pytest.mark.parametrize("dtype", [f32, bf16, u8, tilefish.e4m3])
def test_only_i32_and_i64_carry_a_dimension(dtype):
    with pytest.raises(TilefishDeclarationError, match=r"a runtime scalar that carries a dimension is i32 or i64"):
        dtype[n]


def test_what_is_not_a_dimension_is_refused():
    with pytest.raises(TilefishDeclarationError, match=r"takes a dimension expression"):
        i64["n"]
    with pytest.raises(TilefishDeclarationError, match=r"takes one dimension expression"):
        i64[n, R]


def test_a_static_dimension_or_a_default_is_refused():
    with pytest.raises(TilefishDeclarationError, match="a static dimension; a runtime scalar needs a dynamic Dim"):

        @tilefish.extern(triton="m:k", grid=(1,))
        def k(x: Tensor[n, f32], c: i64[Dim("S", 4)]): ...

    with pytest.raises(TilefishDeclarationError, match="cannot have a default"):

        @tilefish.extern(triton="m:k", grid=(1,))
        def k2(x: Tensor[n, f32], c: i64[n] = 3): ...


def test_only_an_extern_takes_one():
    with pytest.raises(TilefishDeclarationError, match="a 64-bit dimension scalar, which only an extern op takes"):

        @tilefish.kernel
        def k(x: Tensor[n, f32], c: i64[n]) -> Tensor[n, f32]:
            return x

    with pytest.raises(TilefishDeclarationError, match="a 64-bit dimension scalar, which only an extern op takes"):

        @tilefish.library("lib")
        def lib(x: Tensor[n, f32], c: i64[n]) -> Tensor[n, f32]: ...


def test_the_spec_takes_an_i64():
    spec, arg = scalar_arg(wide())
    assert (arg.kind, arg.type, arg.dtype, arg.dim) == ("scalar", "i64", "i64", "n")
    assert spec.signature["n_elem"] == "i64" and spec.schema == 3
    assert (2,) not in spec.attrs  # n is not a multiple of 16
    spec16, arg16 = scalar_arg(wide(dim=n16))
    assert spec16.attrs[(2,)] == [["tt.divisibility", 16]] and arg16.type == "i64"  # a divisible i64 scalar is given the hint, as an i32 one is
    assert any(p.kind == "divisible" and p.arg == "n_elem" for p in spec16.preconditions)
    for op in (wide(), wide(dim=n16)):
        (s,) = compile_specs(op)
        back = CompileSpec.from_json(s.to_json())
        back.verify_against(op)
        assert back.verified


def test_the_type_is_in_the_declaration_digest():
    assert declaration_digest(wide()) != declaration_digest(narrow())
    assert declaration_digest(wide(dtype=i32)) == declaration_digest(narrow())  # i32[n] declares what n does
    assert [s.to_dict() for s in compile_specs(wide(dtype=i32))] == [s.to_dict() for s in compile_specs(narrow())]
    (spec,) = compile_specs(wide())
    with pytest.raises(tilefish.TilefishSpecError, match="first difference"):
        CompileSpec.from_json(spec.to_json()).verify_against(narrow())


def test_the_manifest_records_the_type(tmp_path):
    spec, _ = scalar_arg(wide())
    write_manifest([fake_build(spec)], tmp_path / "manifest.json", cubins=True)
    entry = json.loads((tmp_path / "manifest.json").read_text())["kernels"][0]
    (param,) = [p for p in entry["abi"]["params"] if p["name"] == "n_elem"]
    assert param == {"slot": 2, "index": 2, "name": "n_elem", "kind": "scalar", "type": "i64", "ptx_type": "u64", "size": 8, "dim": "n"}
    Manifest.load(tmp_path).built_kernel(spec, "sm_90", pinned=True)


def test_a_launch_derives_the_value_from_the_shapes_and_checks_its_range():
    """``2**31 + 16`` elements: an ``i64`` count carries it, an ``i32`` one does not (neither kernel has an offset limit: ``index_bits=64``)."""
    big = 2**31 + 16
    x = FakeTensor((big,), "u8", 1 << 44)
    prepared, error = prepare(wide(), [x], next_slab=1 << 50)
    assert error is None
    v = verdict(prepared, error)
    assert v["params"][2] == big and v["grid"] == [-(-big // 1024), 1, 1]
    prepared, error = prepare(narrow(), [x], next_slab=1 << 50)
    assert prepared is None and error.kind == "shape" and error.arg == "n_elem" and "32-bit signed integer" in str(error)
    small = FakeTensor((1000,), "u8", 1 << 44)
    prepared, error = prepare(wide(), [small], next_slab=1 << 50)
    assert error is None and prepared.scalars["n_elem"] == 1000


def test_an_expression_of_dimensions():
    @tilefish.extern(triton="ds_kernels:rows", grid=lambda R: (R,), index_bits=64)
    def rows(x: Tensor[R, C, f32], y: Out[Tensor[R, C, f32]], count: i64[R + C], cols: C): ...

    cols = 2**31 - 1  # the largest C that the i32 scalar `cols` carries: R + C does not fit an i32, and the i64 `count` carries it
    prepared, error = prepare(rows, [FakeTensor((3, cols), "f32", 1 << 44)], next_slab=1 << 50)
    assert error is None and prepared.scalars["count"] == 3 + cols and prepared.scalars["cols"] == cols
    prepared, error = prepare(rows, [FakeTensor((3, cols + 1), "f32", 1 << 44)], next_slab=1 << 50)
    assert prepared is None and error.arg == "cols" and "32-bit signed integer" in str(error)  # each scalar is held to its own type


PTX = """.version 8.8
.target sm_90
.address_size 64

.visible .entry add_one(
\t.param .u64 x,
\t.param .u64 y,
\t.param .u64 n
)
{
\tret;
}
"""


def test_a_prebuilt_kernel_with_a_64_bit_count(tmp_path):
    path = tmp_path / "add_one.ptx"
    path.write_text(PTX)

    def declare(annotation):
        @tilefish.extern(ptx=path, symbol="add_one", block=(256,), grid=lambda n: (tilefish.cdiv(n, 256),), index_bits=64)
        def add_one(x: Tensor[n, u8], y: Out[Tensor[n, u8]], n_elem: annotation): ...

        (spec,) = compile_specs(add_one)
        return spec

    check_ptx_abi(declare(i64[n]), PTX, "add_one")
    with pytest.raises(TilefishBuildError, match=r"does not have the C ABI its stub declares: it takes \['u64', 'u64', 'u64'\]"):
        check_ptx_abi(declare(n), PTX, "add_one")  # an i32 count: not what the PTX takes


def test_a_plan_step_resolves_the_i64_scalar():
    op = wide()

    @tilefish.graph
    def g(x: Tensor[n, u8]):
        return op(x)

    p = tilefish.plan(g, buckets={"n": [1000, 2**31 + 16]})
    from tilefish import _plan

    for values in ({"n": 1000}, {"n": 2**31 + 16}):
        (step,) = p.steps(**values)
        bound = _plan.resolve_step(p.callees[step["entry"]], step["args"], _plan.buffer_shapes(p.doc, values), "test")
        assert bound.scalars["n_elem"] == values["n"]
