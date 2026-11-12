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
    # exact buckets: the bucket value is substituted in the grid, and the static D folds
    assert [s.grid for s in specs] == ["(32, 20)", "(64, 20)", "(128, 20)"]
    assert [s.args[0].shape for s in specs] == [("32", "2560"), ("64", "2560"), ("128", "2560")]
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


# -- argument indices with constexpr and keyword-only parameters before pointers and runtime dims ------------------

nk = Dim("nk", divisible_by=16)


@tilefish.extern(triton="my_kernels:mixed", grid=lambda nk: (nk,))
def mixed_order(
    x: Tensor[nk, bf16],
    *,  # Python needs this: a parameter without a default may not follow one with a default
    B: const = 64,
    y: Out[Tensor[nk, bf16]],
    cnt: nk,
    E: const[f32] = float("-inf"),
    w: InOut[Tensor[nk, f32]],
    t: 2 * nk,
    u: nk + 1,
    v: nk - 16,
): ...


def test_argument_indices_follow_declaration_order_across_constexprs_and_keyword_only_params():
    (spec,) = compile_specs(mixed_order)
    assert list(spec.signature.items()) == [
        ("x", "*bf16"),  # 0
        ("B", "constexpr"),  # 1
        ("y", "*bf16"),  # 2
        ("cnt", "i32"),  # 3
        ("E", "constexpr"),  # 4
        ("w", "*fp32"),  # 5
        ("t", "i32"),  # 6
        ("u", "i32"),  # 7
        ("v", "i32"),  # 8
    ]
    assert [(a.index, a.name, a.kind) for a in spec.args] == [
        (0, "x", "pointer"),
        (1, "B", "constexpr"),
        (2, "y", "pointer"),
        (3, "cnt", "scalar"),
        (4, "E", "constexpr"),
        (5, "w", "pointer"),
        (6, "t", "scalar"),
        (7, "u", "scalar"),
        (8, "v", "scalar"),
    ]
    assert spec.constants == {(1,): 64, (4,): float("-inf")}
    # pointers 0, 2, 5; cnt (nk), t (2*nk) and v (nk - 16) are multiples of 16; u (nk + 1) is not
    assert set(spec.attrs) == {(0,), (2,), (3,), (5,), (6,), (8,)}
    assert all(v == PTR16 for v in spec.attrs.values())
    # launch parameters: the non-constexpr names in order, then the two scratch slots
    assert [p.name for p in spec.launch_params] == ["x", "y", "cnt", "w", "t", "u", "v", "global_scratch", "profile_scratch"]
    assert spec.grid == "(nk,)"
    inp = spec.ast_source_inputs()
    assert list(inp.signature_by_index) == list(range(9))
    assert inp.constants == {(1,): 64, (4,): float("-inf")}


def test_keyword_only_declaration_still_calls_with_the_documented_convention():
    @tilefish.graph
    def g(a: Tensor[nk, bf16], b: InOut[Tensor[nk, f32]]):
        return mixed_order(a, w=b)  # arguments after the * are passed by keyword

    (node,) = g.graph.nodes
    assert node.attrs["inout"] == ("w",)
    assert node.inputs == (g.graph.inputs[0], g.graph.inputs[1])
    assert node.attrs["consts"]["B"] == 64
    assert g.graph.outputs[0] is node.outputs[0]  # y; the new version of w is the last output
    assert len(node.outputs) == 2

    with pytest.raises(tilefish.TilefishTypeError, match="missing a required.*argument: 'w'"):

        @tilefish.graph
        def bad(a: Tensor[nk, bf16], b: InOut[Tensor[nk, f32]]):
            return mixed_order(a)


def test_python_forbids_a_bare_param_after_a_default_so_the_star_is_needed():
    with pytest.raises(SyntaxError):
        compile("def k(x, B=64, y): pass", "<test>", "exec")


# -- bucket semantics are exact -------------------------------------------------------------------------------------------

NH = Dim("NH", 32)
Sb = Dim("S", buckets=[32, 64, 128])


def test_static_dims_fold_in_the_grid_and_in_shapes():
    @tilefish.extern(triton="m:f", grid=lambda S, NH: (S, NH))
    def runtime_s(x: Tensor[Sb, NH, bf16], y: Out[Tensor[Sb, NH, bf16]], s: Sb, nh: const = NH): ...

    (spec,) = compile_specs(runtime_s)  # S is only a runtime argument here, so no expansion
    assert spec.stem == "runtime_s"
    assert spec.grid == "(S, 32)"
    assert spec.args[0].shape == ("S", "32")
    assert spec.constants == {(3,): 32}

    @tilefish.extern(triton="m:f", grid=lambda S, NH: (tilefish.cdiv(S, NH),))
    def cd(x: Tensor[Sb, NH, bf16]): ...

    assert cd.grid_string == "(cdiv(S, 32),)"

    @tilefish.extern(triton="m:f", grid=lambda S, NH: (S, 2 * NH, NH + 0))
    def arith(x: Tensor[Sb, NH, bf16]): ...

    assert arith.grid_string == "(S, 64, 32)"


def test_bucketed_specs_substitute_the_bucket_everywhere_it_is_used():
    @tilefish.extern(triton="m:f", grid=lambda S, NH: (S, NH))
    def a(x: Tensor[Sb, NH, bf16], y: Out[Tensor[Sb, NH, bf16]], nh: const = NH, B: const = Sb): ...

    specs = compile_specs(a)
    assert [s.stem for s in specs] == ["a.S32", "a.S64", "a.S128"]
    assert [s.grid for s in specs] == ["(32, 32)", "(64, 32)", "(128, 32)"]
    assert [s.constants for s in specs] == [{(2,): 32, (3,): 32}, {(2,): 32, (3,): 64}, {(2,): 32, (3,): 128}]
    assert [s.args[0].shape for s in specs] == [("32", "32"), ("64", "32"), ("128", "32")]


def test_grid_and_derived_constants_agree_per_bucket():
    @tilefish.extern(triton="m:f", grid=lambda S, BLOCK: (tilefish.cdiv(S, BLOCK),))
    def b(x: Tensor[Sb, bf16], BLOCK: const = tilefish.cdiv(Sb, 4)): ...

    specs = compile_specs(b)
    assert [s.constants[(1,)] for s in specs] == [8, 16, 32]  # cdiv(S, 4)
    assert [s.grid for s in specs] == ["(4,)", "(4,)", "(4,)"]  # cdiv(S, cdiv(S, 4)), concrete per bucket

    @tilefish.extern(triton="m:f", grid=lambda S, BLOCK: (tilefish.cdiv(S, BLOCK), 1))
    def c(x: Tensor[Sb, bf16], BLOCK: const = 16): ...

    (only,) = compile_specs(c)  # no constexpr is derived from S, so S stays a runtime dimension
    assert only.grid == "(cdiv(S, 16), 1)"


def test_a_bucket_precondition_means_the_dimension_equals_the_bucket():
    @tilefish.extern(triton="m:f", grid=(1,))
    def a(x: Tensor[Sb, bf16], B: const = Sb): ...

    spec = compile_specs(a)[1]
    (bucket,) = [p for p in spec.preconditions if p.kind == "bucket"]
    assert bucket.to_dict() == {"kind": "bucket", "dim": "S", "value": 64, "buckets": [32, 64, 128]}
    assert "exact" in Dim.__doc__ and "pad" in Dim.__doc__  # documented on Dim
    assert "equals" in Precondition.__doc__


def test_bucketed_grid_must_stay_positive():
    @tilefish.extern(triton="m:f", grid=lambda S: (S - 32,))
    def a(x: Tensor[Sb, bf16], B: const = Sb): ...

    with pytest.raises(TilefishSpecError, match="non-positive size 0"):
        compile_specs(a)


# -- JSON: strict, tagged non-finite floats, round trip ---------------------------------------------------------------------


def test_non_finite_floats_become_tagged_values_in_strict_json():
    (spec,) = compile_specs(mixed_order)
    text = spec.to_json()
    doc = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))
    assert doc["constants"] == {"1": 64, "4": {"float": "-inf"}}
    assert doc["args"][4]["value"] == {"float": "-inf"}
    json.dumps(spec.to_dict(), allow_nan=False)  # strict
    assert "Infinity" not in text and "NaN" not in text

    @tilefish.extern(triton="m:f", grid=(1,))
    def special(x: Tensor[nk, bf16], A: const[f32] = float("inf"), B: const[f32] = float("nan"), C: const[f32] = -1.5): ...

    (s2,) = compile_specs(special)
    doc2 = json.loads(s2.to_json())
    assert doc2["constants"] == {"1": {"float": "inf"}, "2": {"float": "nan"}, "3": -1.5}


def test_specs_round_trip_through_dict_and_json():
    import math

    specs = [*compile_specs(mixed_order), *compile_specs(gelu), *compile_specs(rmsnorm), *compile_specs(qkv_rope)]
    assert len(specs) == 6
    for spec in specs:
        assert CompileSpec.from_json(spec.to_json()) == spec
        assert CompileSpec.from_dict(spec.to_dict()) == spec
        assert CompileSpec.from_json(spec.to_json(indent=None)).to_dict() == spec.to_dict()
    back = CompileSpec.from_json(compile_specs(mixed_order)[0].to_json())
    assert back.constants == {(1,): 64, (4,): float("-inf")}
    assert back.attrs[(0,)] == PTR16 and isinstance(back.block, tuple)
    assert back.launch_params[-1].value == 0
    back.verify_against(mixed_order)  # a spec read back must be verified before it feeds triton.compile
    assert back.ast_source_inputs().constants == {(1,): 64, (4,): float("-inf")}

    @tilefish.extern(triton="m:f", grid=(1,))
    def with_nan(x: Tensor[nk, bf16], A: const[f32] = float("nan")): ...

    (s,) = compile_specs(with_nan)
    assert math.isnan(CompileSpec.from_json(s.to_json()).constants[(1,)])


def test_from_dict_rejects_an_unknown_schema():
    doc = compile_specs(gelu)[0].to_dict()
    doc["schema"] = 99
    with pytest.raises(TilefishSpecError, match="unsupported compile spec schema 99"):
        CompileSpec.from_dict(doc)


# -- launch parameters of the extern declaration --------------------------------------------------------------------------


@pytest.mark.parametrize("warps", [1, 2, 4, 8, 16, 32])
def test_valid_num_warps(warps):
    @tilefish.extern(triton="m:f", num_warps=warps, grid=(1,))
    def e(x: Tensor[nk, bf16]): ...

    assert compile_specs(e)[0].block == (32 * warps, 1, 1)


@pytest.mark.parametrize("warps", [0, 3, 5, 6, 7, 12, 33, 64, -4, True, 4.0, "4", None])
def test_invalid_num_warps(warps):
    with pytest.raises(tilefish.TilefishDeclarationError, match="num_warps must be a power of two from 1 to 32"):
        tilefish.extern(triton="m:f", num_warps=warps, grid=(1,))


@pytest.mark.parametrize("grid", [(0,), (-1,), (4, 0), (2, 2, -3), lambda nk: (nk - nk,)])
def test_static_grid_sizes_must_be_at_least_one(grid):
    with pytest.raises(tilefish.TilefishDeclarationError, match="grid sizes must be at least 1"):

        @tilefish.extern(triton="m:f", grid=grid)
        def e(x: Tensor[nk, bf16]): ...
