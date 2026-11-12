import json
import types

import pytest

import tilefish
from tilefish import (
    BlockScaled,
    Dim,
    InOut,
    Out,
    Tensor,
    TilefishSpecError,
    bf16,
    const,
    e4m3,
    f32,
    i32,
    ue8m0,
)
from tilefish.aot import CompileSpec, Precondition, compile_specs

M = Dim("M")
D = Dim("D", 2560)
n = Dim("n", divisible_by=16)

PTR16 = [["tt.divisibility", 16]]


# (a) a norm kernel (X, W, Y, N, BLOCK, EPS) with three constexprs ----------------------------------


@tilefish.extern(triton="my_kernels:rmsnorm_fwd", num_warps=8, grid=lambda M: (M,))
def rmsnorm(
    X: Tensor[M, D, bf16],
    W: Tensor[D, bf16],
    Y: Out[Tensor[M, D, bf16]],
    N: const = D,
    BLOCK: const = 4096,
    EPS: const[f32] = 1e-6,
): ...


def test_norm_kernel_spec():
    (spec,) = compile_specs(rmsnorm)
    assert isinstance(spec, CompileSpec)
    assert spec.stem == "rmsnorm"
    assert spec.op == "rmsnorm"
    assert spec.triton == "my_kernels:rmsnorm_fwd"
    assert (spec.triton_module, spec.triton_function) == ("my_kernels", "rmsnorm_fwd")
    assert spec.signature == {
        "X": "*bf16",
        "W": "*bf16",
        "Y": "*bf16",
        "N": "constexpr",
        "BLOCK": "constexpr",
        "EPS": "constexpr",
    }
    assert list(spec.signature) == ["X", "W", "Y", "N", "BLOCK", "EPS"]  # declaration order
    assert spec.constants == {(3,): 2560, (4,): 4096, (5,): 1e-6}
    assert isinstance(spec.constants[(5,)], float)
    assert spec.attrs == {(0,): PTR16, (1,): PTR16, (2,): PTR16}
    assert (spec.num_warps, spec.num_stages) == (8, 3)
    assert spec.grid == "(M,)"
    assert spec.block == (256, 1, 1)
    assert [p.name for p in spec.launch_params] == ["X", "W", "Y", "global_scratch", "profile_scratch"]
    assert [(p.type, p.value) for p in spec.launch_params[-2:]] == [("u64", 0), ("u64", 0)]


def test_norm_kernel_preconditions():
    (spec,) = compile_specs(rmsnorm)
    assert [p.to_dict() for p in spec.preconditions] == [
        {"kind": "pointer_alignment", "arg": "X", "index": 0, "bytes": 16},
        {"kind": "static_size", "arg": "X", "index": 0, "axis": 1, "value": 2560},
        {"kind": "pointer_alignment", "arg": "W", "index": 1, "bytes": 16},
        {"kind": "static_size", "arg": "W", "index": 1, "axis": 0, "value": 2560},
        {"kind": "pointer_alignment", "arg": "Y", "index": 2, "bytes": 16},
        {"kind": "static_size", "arg": "Y", "index": 2, "axis": 1, "value": 2560},
    ]


# (b) a gelu kernel (X, Y, n_elem, BLOCK) with n_elem divisible_by=16 -------------------------------


@tilefish.extern(triton="my_kernels:gelu_kernel", num_warps=4, grid=lambda n: (tilefish.cdiv(n, 1024),))
def gelu(X: Tensor[n, bf16], Y: Out[Tensor[n, bf16]], n_elem: n, BLOCK: const = 1024): ...


def test_gelu_kernel_spec():
    (spec,) = compile_specs(gelu)
    assert spec.signature == {"X": "*bf16", "Y": "*bf16", "n_elem": "i32", "BLOCK": "constexpr"}
    assert spec.constants == {(3,): 1024}
    assert spec.attrs == {(0,): PTR16, (1,): PTR16, (2,): PTR16}  # n_elem divisible by 16: index 2
    assert spec.grid == "(cdiv(n, 1024),)"
    assert spec.block == (128, 1, 1)
    assert [p.name for p in spec.launch_params] == ["X", "Y", "n_elem", "global_scratch", "profile_scratch"]
    assert [(p.name, p.type) for p in spec.launch_params[:3]] == [("X", "*bf16"), ("Y", "*bf16"), ("n_elem", "i32")]
    assert {"kind": "divisible", "arg": "n_elem", "index": 2, "dim": "n", "by": 16} in [
        p.to_dict() for p in spec.preconditions
    ]


def test_i32_arg_without_divisibility_gets_no_attr():
    plain = Dim("p")

    @tilefish.extern(triton="my_kernels:g", grid=lambda p: (p,))
    def g(X: Tensor[plain, bf16], Y: Out[Tensor[plain, bf16]], n_elem: plain): ...

    (spec,) = compile_specs(g)
    assert spec.attrs == {(0,): PTR16, (1,): PTR16}  # no attr at index 2
    assert not any(p.kind == "divisible" for p in spec.preconditions)


def test_divisibility_attr_needs_a_multiple_of_16():
    d8 = Dim("d8", divisible_by=8)
    d128 = Dim("d128", divisible_by=128)

    @tilefish.extern(triton="my_kernels:g", grid=(1,))
    def g(X: Tensor[d8, d128, bf16], a: d8, b: d128, c: 2 * d8): ...

    (spec,) = compile_specs(g)
    assert spec.signature == {"X": "*bf16", "a": "i32", "b": "i32", "c": "i32"}
    # d8 is divisible by 8 only; 2*d8 is divisible by 16; d128 by 128; the attr value is always 16
    assert spec.attrs == {(0,): PTR16, (2,): PTR16, (3,): PTR16}


# (c) a bucketed constexpr S over 3 bucket values ---------------------------------------------------

S = Dim("S", buckets=[32, 64, 128])


@tilefish.extern(triton="my_kernels:qkv_rope", num_warps=4, num_stages=2, grid=lambda S, D: (S, tilefish.cdiv(D, 128)))
def qkv_rope(X: Tensor[S, D, bf16], Y: Out[Tensor[S, D, bf16]], BLOCK_S: const = S, BLOCK_D: const = 128): ...


def test_bucketed_constexpr_expands_to_one_spec_per_bucket():
    specs = compile_specs(qkv_rope)
    assert [s.stem for s in specs] == ["qkv_rope.S32", "qkv_rope.S64", "qkv_rope.S128"]
    assert [s.constants for s in specs] == [
        {(2,): 32, (3,): 128},
        {(2,): 64, (3,): 128},
        {(2,): 128, (3,): 128},
    ]
    assert [s.buckets for s in specs] == [{"S": 32}, {"S": 64}, {"S": 128}]
    for s in specs:
        assert s.signature == {"X": "*bf16", "Y": "*bf16", "BLOCK_S": "constexpr", "BLOCK_D": "constexpr"}
        assert s.attrs == {(0,): PTR16, (1,): PTR16}
        assert (s.num_warps, s.num_stages) == (4, 2)
        assert s.grid == "(S, 20)"  # the static D folds: cdiv(2560, 128)
    bucket = [p.to_dict() for p in specs[1].preconditions if p.kind == "bucket"]
    assert bucket == [{"kind": "bucket", "dim": "S", "value": 64, "buckets": [32, 64, 128]}]


def test_several_bucketed_dims_expand_to_a_product():
    T = Dim("T", buckets=[1, 2])

    @tilefish.extern(triton="my_kernels:k", grid=(1,))
    def k(X: Tensor[S, T, bf16], A: const = S, B: const = T, C: const = S + T): ...

    specs = compile_specs(k)
    assert [s.stem for s in specs] == [
        "k.S32.T1",
        "k.S32.T2",
        "k.S64.T1",
        "k.S64.T2",
        "k.S128.T1",
        "k.S128.T2",
    ]
    assert specs[1].constants == {(1,): 32, (2,): 2, (3,): 34}


def test_bucketed_dim_that_is_only_a_runtime_argument_does_not_expand():
    @tilefish.extern(triton="my_kernels:k", grid=lambda S: (S,))
    def k(X: Tensor[S, bf16], s: S): ...

    (spec,) = compile_specs(k)
    assert spec.stem == "k"
    assert spec.signature == {"X": "*bf16", "s": "i32"}
    assert spec.attrs == {(0,): PTR16}  # nothing declares S divisible, buckets do not imply it


# -- spec contents ----------------------------------------------------------------------------------


def test_ast_source_inputs_are_exactly_what_triton_takes():
    (spec,) = compile_specs(gelu)
    inp = spec.ast_source_inputs()
    assert inp.signature == {"X": "*bf16", "Y": "*bf16", "n_elem": "i32", "BLOCK": "constexpr"}
    assert inp.attrs == {(0,): PTR16, (1,): PTR16, (2,): PTR16}
    assert inp.constants == {(3,): 1024}
    assert inp.options == {"num_warps": 4, "num_stages": 3}
    assert inp.signature_by_index == {0: "*bf16", 1: "*bf16", 2: "i32", 3: "constexpr"}
    # fresh copies: mutating them does not change the spec
    inp.attrs[(0,)][0][1] = 1
    inp.constants[(3,)] = 0
    inp.signature["X"] = "*fp32"
    again = spec.ast_source_inputs()
    assert again.attrs[(0,)] == PTR16 and again.constants == {(3,): 1024} and again.signature["X"] == "*bf16"


def test_ast_source_inputs_drive_an_astsource_like_call():
    """The inputs plug into ``ASTSource(fn, signature=..., constexprs=None, attrs=...)`` unchanged."""
    seen = {}

    class FakeASTSource:  # same constructor shape as triton.compiler.ASTSource
        def __init__(self, fn, signature, constexprs=None, attrs=None):
            seen.update(fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
            self.constants = {}

    (spec,) = compile_specs(gelu)
    inp = spec.ast_source_inputs()
    src = FakeASTSource("fn", signature=inp.signature, constexprs=None, attrs=inp.attrs)
    src.constants = inp.constants
    assert seen["signature"] is inp.signature and seen["attrs"] is inp.attrs
    assert src.constants == {(3,): 1024}
    assert all(isinstance(k, str) for k in inp.signature) and all(isinstance(k, tuple) for k in inp.constants)


def test_to_json_manifest_skeleton():
    (spec,) = compile_specs(gelu)
    doc = json.loads(spec.to_json())
    assert doc["stem"] == "gelu"
    assert doc["triton"] == {"module": "my_kernels", "function": "gelu_kernel", "ref": "my_kernels:gelu_kernel"}
    assert doc["signature"] == {"X": "*bf16", "Y": "*bf16", "n_elem": "i32", "BLOCK": "constexpr"}
    assert list(doc["signature"]) == ["X", "Y", "n_elem", "BLOCK"]
    assert doc["constants"] == {"3": 1024}
    assert doc["attrs"] == {"0": PTR16, "1": PTR16, "2": PTR16}
    assert doc["grid"] == "(cdiv(n, 1024),)"
    assert doc["block"] == [128, 1, 1]
    assert doc["num_warps"] == 4 and doc["num_stages"] == 3
    assert doc["launch_params"][-2:] == [
        {"name": "global_scratch", "type": "u64", "value": 0},
        {"name": "profile_scratch", "type": "u64", "value": 0},
    ]
    assert doc["args"][2] == {"index": 2, "name": "n_elem", "kind": "scalar", "type": "i32", "dtype": "i32", "dim": "n"}
    assert doc["args"][0]["shape"] == ["n"] and doc["args"][0]["access"] == "in"
    assert doc["args"][1]["access"] == "out"
    assert doc["args"][3] == {"index": 3, "name": "BLOCK", "kind": "constexpr", "type": "constexpr", "value": 1024}
    assert {"kind": "pointer_alignment", "arg": "X", "index": 0, "bytes": 16} in doc["preconditions"]
    for forbidden in ("cubin", "sha256", "hash", "digest"):
        assert forbidden not in doc
    assert "schema" in doc


def test_bucketed_json_records_the_bucket():
    doc = json.loads(compile_specs(qkv_rope)[2].to_json(indent=None))
    assert doc["stem"] == "qkv_rope.S128" and doc["buckets"] == {"S": 128}


def test_pointer_dtypes_in_the_signature():
    sdim = Dim("sd")

    @tilefish.extern(triton="my_kernels:mixed", grid=(1,))
    def mixed(
        a: Tensor[sdim, tilefish.f16],
        b: Tensor[sdim, f32],
        c: Tensor[sdim, tilefish.i32],
        d: Tensor[sdim, tilefish.i64],
        e: Tensor[sdim, tilefish.u8],
        f: Tensor[sdim, e4m3],
        g: Tensor[sdim, tilefish.e5m2],
        h: Tensor[sdim, tilefish.e2m1],
        i: Tensor[sdim, ue8m0],
        j: InOut[Tensor[sdim, bf16]],
    ): ...

    (spec,) = compile_specs(mixed)
    assert list(spec.signature.values()) == [
        "*fp16", "*fp32", "*i32", "*i64", "*u8", "*fp8e4nv", "*fp8e5", "*u8", "*u8", "*bf16",
    ]
    assert spec.args[9].access == "inout"


def test_block_scaled_tensors_are_not_supported_yet():
    K32 = Dim("K32", divisible_by=32)

    @tilefish.extern(triton="my_kernels:q", grid=(1,))
    def q(w: Tensor[M, K32, BlockScaled(e4m3, ue8m0, 32)]): ...

    with pytest.raises(TilefishSpecError, match="block-scaled tensors"):
        compile_specs(q)


def test_triton_reference_from_an_object():
    def kernel_fn():
        pass

    kernel_fn.__module__ = "my_kernels"
    ns = types.SimpleNamespace(gelu_kernel=kernel_fn)

    @tilefish.extern(triton=ns.gelu_kernel, grid=lambda M: (M,))
    def e(X: Tensor[M, bf16], Y: Out[Tensor[M, bf16]]): ...

    assert e.triton == "my_kernels:kernel_fn"

    class JitLike:  # triton.JITFunction keeps the python function in .fn
        fn = kernel_fn

    @tilefish.extern(triton=JitLike(), grid=lambda M: (M,))
    def e2(X: Tensor[M, bf16], Y: Out[Tensor[M, bf16]]): ...

    assert e2.triton == "my_kernels:kernel_fn"

    with pytest.raises(tilefish.TilefishDeclarationError, match="module:function"):
        tilefish.extern(triton=types.SimpleNamespace(), grid=(1,))


def test_grid_forms():
    pn = Dim("pn")

    def stub(**kw):
        @tilefish.extern(triton="m:f", **kw)
        def e(X: Tensor[pn, M, bf16], BLOCK: const = 256): ...

        return e

    assert stub(grid=(4, 2, 1)).grid_string == "(4, 2, 1)"
    assert stub(grid=(tilefish.cdiv(pn, 128),)).grid_string == "(cdiv(pn, 128),)"
    assert stub(grid=lambda pn, M: (pn, M)).grid_string == "(pn, M)"
    assert stub(grid=lambda pn, BLOCK: (tilefish.cdiv(pn, BLOCK),)).grid_string == "(cdiv(pn, 256),)"
    assert stub(grid=lambda pn: (2 * pn + 1, 1)).grid_string == "(2*pn + 1, 1)"
    assert stub(grid=lambda pn: [pn]).grid_string == "(pn,)"
    assert stub(grid=None).grid is None
    with pytest.raises(tilefish.TilefishDeclarationError, match="tuple of 1 to 3 sizes"):
        stub(grid=(1, 2, 3, 4))
    with pytest.raises(tilefish.TilefishDeclarationError, match="ints or dimension expressions"):
        stub(grid=("x",))


def test_grid_is_required_for_specs_and_only_extern_ops_have_them():
    @tilefish.extern(triton="m:f")
    def nogrid(X: Tensor[M, bf16]): ...

    with pytest.raises(TilefishSpecError, match="has no grid"):
        compile_specs(nogrid)
    with pytest.raises(TilefishSpecError, match="takes an extern op"):
        compile_specs(lambda: None)


def test_const_default_that_depends_on_an_unbucketed_dynamic_dim():
    @tilefish.extern(triton="m:f", grid=(1,))
    def bad(X: Tensor[M, bf16], B: const = M): ...

    with pytest.raises(TilefishSpecError, match="dynamic dimension M that has no buckets"):
        compile_specs(bad)


def test_const_defaults_of_every_kind():
    @tilefish.extern(triton="m:f", grid=(1,))
    def e(X: Tensor[M, bf16], A: const = 3, B: const = 0.5, C: const = True, E: const[i32] = 7, F_: const = 2 * D): ...

    (spec,) = compile_specs(e)
    assert spec.constants == {(1,): 3, (2,): 0.5, (3,): True, (4,): 7, (5,): 5120}
    assert spec.signature["C"] == "constexpr"


def test_precondition_to_dict_omits_unset_fields():
    assert Precondition("static_size", arg="w", index=1, axis=0, value=8).to_dict() == {
        "kind": "static_size",
        "arg": "w",
        "index": 1,
        "axis": 0,
        "value": 8,
    }
