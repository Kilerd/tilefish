"""``Strided[...]`` and ``StrideOf`` parameters of an extern: declaration, calls in a trace, the compile spec, the ABI and the launch checks.

A ``Strided[Tensor[...]]`` parameter takes a view: the kernel gets the pointer ``base + offset * itemsize`` and the strides of the view as the scalar
parameters the stub declares with ``StrideOf("x", axis)``, in the kernel's argument order. The strides and the offset are dimensions of the signature:
a call binds them like it binds the shape. ``Strided[..., 16]`` says the pointer is compiled as 16-byte aligned (``tt.divisibility=16``), which the
offset must then provably allow. Nothing here needs a GPU.
"""

import copy

import pytest
from fake_torch import FakeTensor, prepare

import tilefish
from tilefish import (
    DevScalar,
    Dim,
    Out,
    Strided,
    StrideOf,
    Tensor,
    TilefishDeclarationError,
    TilefishPreconditionError,
    TilefishSpecError,
    TilefishTypeError,
    bf16,
    const,
    f32,
    i32,
)
from tilefish._build import abi_of
from tilefish._launch import canonical_strides, extent_elements
from tilefish.aot import CompileSpec, compile_specs, declaration_digest

S = Dim("S")
H = 8


@tilefish.extern(triton="kernels:head_copy", num_warps=1, grid=lambda S: (S, 8))
def head_copy(
    x: Strided[Tensor[S, 8, 64, bf16]], y: Out[Tensor[S, 8, 64, bf16]], x_s0: StrideOf("x", 0), x_s1: StrideOf("x", 1), D: const = 64
): ...


@tilefish.extern(triton="kernels:head_copy16", num_warps=1, grid=lambda S: (S, 8))
def head_copy16(
    x: Strided[Tensor[S, 8, 64, bf16], 16],
    y: Out[Tensor[S, 8, 64, bf16]],
    x_s0: StrideOf("x", 0, divisible_by=16),
    x_s1: StrideOf("x", 1),
    D: const = 64,
): ...


@tilefish.extern(triton="kernels:head_copy64", num_warps=1, grid=lambda S: (S, 8), index_bits=64)
def head_copy64(x: Strided[Tensor[S, 8, 64, bf16]], y: Out[Tensor[S, 8, 64, bf16]], x_s0: StrideOf("x", 0), x_s1: StrideOf("x", 1)): ...


@tilefish.extern(triton="kernels:plain", grid=lambda S: (S,))
def plain(x: Tensor[S, 64, bf16], y: Out[Tensor[S, 64, bf16]]): ...


# -- declarations ---------------------------------------------------------------------------------------------------------------------------


def test_a_strided_parameter_has_the_strides_and_the_offset_its_stub_declares():
    p = head_copy.signature.params[0]
    assert p.strided and p.align is None and not p.type.contiguous
    assert [str(s) for s in p.type.strides] == ["x_s0", "x_s1", "1"] and str(p.type.offset) == "x_offset"
    assert p.type.offset.divisible_by is None
    q = head_copy16.signature.params[0]
    assert q.align == 16 and q.type.offset.divisible_by == 8  # the elements in 16 bytes of bf16
    assert head_copy16.signature.params[2].dim.divisible_by == 16 and head_copy16.signature.params[2].stride_of == ("x", 0)
    assert str(head_copy) == (
        "extern head_copy(x: Strided[Tensor[S, 8, 64, bf16]], y: Out[Tensor[S, 8, 64, bf16]], x_s0: StrideOf('x', 0), x_s1: StrideOf('x', 1), D: const = 64)"
    )
    assert [p.name for p in head_copy.signature.params if p.kind == "dim"] == ["x_s0", "x_s1"]
    assert all(not p.is_callable_arg for p in head_copy.signature.params if p.kind == "dim")  # a caller never passes a stride


def test_only_the_last_axis_may_go_without_a_stride_parameter():
    with pytest.raises(TilefishDeclarationError, match=r"has no StrideOf\('x', 0\) parameter: only the last axis may go without one"):

        @tilefish.extern(triton="m:f", grid=(1,))
        def f(x: Strided[Tensor[4, 8, bf16]], x_s1: StrideOf("x", 1)): ...

    @tilefish.extern(triton="m:g", grid=(1,))
    def g(x: Strided[Tensor[8, bf16]], y: Out[Tensor[8, bf16]]): ...  # rank 1: the only axis is the last one, with stride 1

    assert [str(s) for s in g.signature.params[0].type.strides] == ["1"]


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: Strided[3], "wraps a tensor type"),
        (lambda: Strided[Tensor[4, bf16], 8], "the alignment must be 16"),
        (lambda: Strided[DevScalar[i32]], "only plain tensors can be strided"),
        (lambda: Strided[Tensor[4, 32, tilefish.BlockScaled(tilefish.e4m3, tilefish.ue8m0, 32)]], "only plain tensors can be strided"),
        (lambda: tilefish.InOut[Strided[Tensor[4, bf16]]], "only input tensors can be strided"),
        (lambda: StrideOf(3, 0), "names a tensor parameter"),
        (lambda: StrideOf("x", -1), "non-negative int"),
        (lambda: StrideOf("x", 0, divisible_by=0), "positive int"),
    ],
)
def test_the_annotations_check_themselves(build, message):
    with pytest.raises(TilefishDeclarationError, match=message):
        build()


def test_a_stride_parameter_must_belong_to_a_strided_tensor_of_the_same_op():
    with pytest.raises(TilefishDeclarationError, match=r"x_s0 is StrideOf\('x', 0\), but 'x' is not a Strided"):

        @tilefish.extern(triton="m:f", grid=(1,))
        def f(x: Tensor[4, 8, bf16], x_s0: StrideOf("x", 0)): ...

    with pytest.raises(TilefishDeclarationError, match=r"x_s5 is the stride of axis 5 of 'x', which has rank 2"):

        @tilefish.extern(triton="m:f", grid=(1,))
        def f2(x: Strided[Tensor[4, 8, bf16]], x_s5: StrideOf("x", 5)): ...

    with pytest.raises(TilefishDeclarationError, match=r"declared twice \(a and b\)"):

        @tilefish.extern(triton="m:f", grid=(1,))
        def f3(x: Strided[Tensor[4, 8, bf16]], a: StrideOf("x", 0), b: StrideOf("x", 0)): ...

    with pytest.raises(TilefishDeclarationError, match="cannot have a default"):

        @tilefish.extern(triton="m:f", grid=(1,))
        def f4(x: Strided[Tensor[4, 8, bf16]], a: StrideOf("x", 0) = 3): ...

    with pytest.raises(TilefishDeclarationError, match="clashes with the offset dimension"):

        @tilefish.extern(triton="m:f", grid=(1,))
        def f5(x: Strided[Tensor[4, 8, bf16]], x_offset: Dim("q"), a: StrideOf("x", 0)): ...


def test_strided_and_stride_parameters_are_for_externs_only():
    with pytest.raises(TilefishDeclarationError, match="only an extern op can take"):

        @tilefish.graph
        def g(x: Strided[Tensor[4, 8, bf16]]):
            return x

    with pytest.raises(TilefishDeclarationError, match="only an extern op can take"):

        @tilefish.kernel
        def k(x: Tensor[4, 8, bf16], s: StrideOf("x", 0)) -> Tensor[4, 8, bf16]:
            return x

    with pytest.raises(TilefishDeclarationError, match="DevScalar, which a library op cannot take"):

        @tilefish.library("lib")
        def lib(x: Tensor[4, bf16], n: DevScalar[i32]) -> Tensor[4, bf16]: ...


# -- calls in a trace -----------------------------------------------------------------------------------------------------------------------


def test_a_view_is_passed_to_a_strided_parameter_and_binds_its_layout():
    @tilefish.graph
    def g(qkv: Tensor[S, 3, 8, 64, bf16]):
        return head_copy(qkv[:, 1]), head_copy(qkv[:, 2]), head_copy(qkv[:, 0])

    assert all(str(t) == "Tensor[S, 8, 64, bf16]" for t in (v.type for v in g.graph.outputs))
    assert [n.op for n in g.graph.nodes] == ["select", "head_copy", "select", "head_copy", "select", "head_copy"]


def test_a_dense_tensor_and_a_transposed_one_are_views_too():
    @tilefish.graph
    def g(x: Tensor[S, 8, 64, bf16], t: Tensor[8, S, 64, bf16]):
        return head_copy(x), head_copy(t.permute(1, 0, 2))

    assert len(g.graph.outputs) == 2


def test_the_last_axis_of_the_view_must_have_stride_one():
    """No trace operation makes a view with a step on its last axis, so the equation a call solves is built from a type."""
    from tilefish.signature import _add_layout_equations, _Unifier
    from tilefish.trace import TraceContext
    from tilefish.types import TensorType

    p = head_copy.signature.params[0]
    unifier = _Unifier(head_copy, TraceContext("t", "graph"))
    _add_layout_equations(head_copy, p, p.type, TensorType([S, 8, 64], bf16, strides=[1024, 128, 2]), unifier)
    with pytest.raises(TilefishTypeError, match=r"the stride of axis 2 is 2, but the signature requires 1 \(the kernel loads along that axis"):
        unifier.solve()
    ok = _Unifier(head_copy, TraceContext("t", "graph"))
    _add_layout_equations(head_copy, p, p.type, TensorType([S, 8, 64], bf16, strides=[1024, 128, 1], offset=64), ok)
    ok.solve()
    assert {d.name: str(v) for d, v in ok.bindings.items()} == {"x_s0": "1024", "x_s1": "128", "x_offset": "64"}


def test_a_view_passed_to_a_parameter_that_takes_a_tensor_says_how_to_fix_it():
    with pytest.raises(
        TilefishTypeError,
        match=r"plain: argument 'x' is a strided view \(View\[S, 64, bf16\]\); this parameter takes a contiguous tensor: pass the unsplit tensor, "
        r"compute from the view \(x \+ 0 is a contiguous copy\), or have an extern declare the parameter Strided\[\.\.\.\]",
    ):

        @tilefish.graph
        def g(x: Tensor[S, 128, bf16]):
            return plain(x.split(64, axis=1)[0])

    @tilefish.graph
    def fixed(x: Tensor[S, 128, bf16]):
        return plain(x.split(64, axis=1)[0] + 0)

    assert len(fixed.graph.outputs) == 1


@tilefish.extern(triton="kernels:five", grid=(1,))
def five(x: Strided[Tensor[5, bf16], 16], y: Out[Tensor[5, bf16]]): ...


@tilefish.extern(triton="kernels:sixteen", grid=(1,))
def sixteen(x: Strided[Tensor[16, bf16], 16], y: Out[Tensor[16, bf16]]): ...


def test_a_view_that_does_not_provably_start_on_16_bytes_is_refused_by_strided_16():
    @tilefish.graph
    def fine(qkv: Tensor[S, 3, 8, 64, bf16], t: Tensor[4, 16, bf16], u: Tensor[4, 5, bf16]):
        return head_copy16(qkv[:, 1]), sixteen(t[1]), five(u[0])  # 1024 bytes in; 32 bytes in; at the start

    assert len(fine.graph.outputs) == 3
    with pytest.raises(
        TilefishTypeError,
        match=r"five: argument 'x': the offset of the view is 5 elements, which is not a multiple of 8 \(16 bytes\); "
        r"this parameter is Strided\[\.\.\., 16\] and its kernel is compiled for a 16-byte aligned pointer",
    ):

        @tilefish.graph
        def odd(u: Tensor[4, 5, bf16]):
            return five(u[1])  # 10 bytes in

    # Strided without the 16 takes any offset
    @tilefish.extern(triton="kernels:five_any", grid=(1,))
    def five_any(x: Strided[Tensor[5, bf16]], y: Out[Tensor[5, bf16]]): ...

    @tilefish.graph
    def anything(u: Tensor[4, 5, bf16]):
        return five_any(u[1]), five_any(u[3])

    assert len(anything.graph.outputs) == 2


def test_an_offset_that_depends_on_a_dimension_is_a_requirement_for_the_host_to_check():
    a, b = Dim("a"), Dim("b")

    @tilefish.extern(triton="kernels:rows", grid=(1,))
    def rows(x: Strided[Tensor[b, 8, bf16], 16], y: Out[Tensor[b, 8, bf16]], x_s0: StrideOf("x", 0)): ...

    @tilefish.graph
    def g(x: Tensor[a + b, 8, bf16]):
        first, second = x.split([a, b], axis=0)
        return rows(second)  # starts 8a elements in: a multiple of 8, whatever a is

    assert g.graph.requirements == ()

    @tilefish.extern(triton="kernels:rows3", grid=(1,))
    def rows3(x: Strided[Tensor[b, 3, bf16], 16], y: Out[Tensor[b, 3, bf16]], x_s0: StrideOf("x", 0)): ...

    @tilefish.graph
    def h(x: Tensor[a + b, 3, bf16]):
        first, second = x.split([a, b], axis=0)
        return rows3(second)  # 3a elements in: only if a is a multiple of 8

    assert [str(r) for r in h.graph.requirements] == [
        "(3*a) % 8 == 0  (required by rows3: argument 'x')"
    ]


# -- the compile spec -----------------------------------------------------------------------------------------------------------------------


def spec_of(op):
    (spec,) = compile_specs(op)
    return spec


def test_a_strided_pointer_records_its_strides_and_offset_and_a_stride_scalar_its_origin():
    spec = spec_of(head_copy)
    x, y, s0, s1, D = spec.args
    assert (x.kind, x.strides, x.offset, x.shape_dims) == ("pointer", ("x_s0", "x_s1", 1), "x_offset", ("S", 8, 64))
    assert y.strides is None and y.offset is None  # an Out tensor is dense
    assert [(a.name, a.kind, a.type, a.dim, a.stride_of) for a in (s0, s1)] == [
        ("x_s0", "scalar", "i32", "x_s0", ("x", 0)),
        ("x_s1", "scalar", "i32", "x_s1", ("x", 1)),
    ]
    assert spec.signature == {"x": "*bf16", "y": "*bf16", "x_s0": "i32", "x_s1": "i32", "D": "constexpr"}
    assert spec.dims == {"S": {}, "x_s0": {}, "x_s1": {}, "x_offset": {}}  # the names a host binds
    assert spec.schema == 3 and spec.uses_strided_schema and spec.launch_params[0].name == "x" and spec.launch_params[-1].name == "profile_scratch"


def test_the_pointer_alignment_follows_from_the_offset():
    free = spec_of(head_copy)
    aligned = spec_of(head_copy16)
    assert (0,) not in free.attrs and free.attrs == {(1,): [["tt.divisibility", 16]]}  # no hint for the view, one for the dense output
    assert aligned.attrs[(0,)] == [["tt.divisibility", 16]] and aligned.attrs[(2,)] == [["tt.divisibility", 16]]  # and the stride that is a multiple of 16
    alignment = {p.arg: p.bytes for p in free.preconditions if p.kind == "pointer_alignment"}
    assert alignment == {"x": 2, "y": 16}  # a bf16 element, and the 16 bytes every other pointer promises
    assert {p.arg: p.bytes for p in aligned.preconditions if p.kind == "pointer_alignment"} == {"x": 16, "y": 16}
    assert aligned.dims["x_offset"] == {"divisible_by": 8} and aligned.dims["x_s0"] == {"divisible_by": 16}
    assert any(p.kind == "divisible" and p.arg == "x_s0" and p.by == 16 for p in aligned.preconditions)


def test_a_strided_pointer_has_a_max_extent_and_a_dense_one_a_max_elements():
    spec = spec_of(head_copy)
    kinds = {(p.kind, p.arg) for p in spec.preconditions if p.kind in ("max_elements", "max_extent")}
    assert kinds == {("max_extent", "x"), ("max_elements", "y")}
    (extent,) = [p for p in spec.preconditions if p.kind == "max_extent"]
    assert extent.to_dict() == {"kind": "max_extent", "arg": "x", "index": 0, "value": 2**31 - 1}
    wide = spec_of(head_copy64)
    assert not [p for p in wide.preconditions if p.kind in ("max_elements", "max_extent")]  # index_bits=64: no limit
    assert wide.signature["x_s0"] == "i64" and wide.args[2].type == "i64" and wide.args[2].dtype == "i64" and wide.launch_params[2].type == "i64"


def test_the_abi_of_a_strided_kernel():
    abi = abi_of(spec_of(head_copy))
    x, y, s0, s1 = abi["params"]
    assert x["strides"] == ["x_s0", "x_s1", 1] and x["offset"] == "x_offset" and x["shape"] == ["S", 8, 64]
    assert "strides" not in y and "offset" not in y
    assert (s0["kind"], s0["type"], s0["ptx_type"], s0["size"], s0["dim"], s0["stride_of"]) == ("scalar", "i32", "u32", 4, "x_s0", {"arg": "x", "axis": 0})
    assert abi["scratch_slots"] == 2 and len(abi["scratch"]) == 2
    wide = abi_of(spec_of(head_copy64))["params"][2]
    assert (wide["type"], wide["ptx_type"], wide["size"]) == ("i64", "u64", 8)
    assert abi_of(spec_of(plain))["params"][0].keys().isdisjoint({"strides", "offset"})  # an ordinary pointer says nothing of them


def test_specs_that_use_nothing_new_are_schema_2_and_their_digests_are_unchanged():
    """The declaration digests and the spec fingerprints of ops that use none of the new features are what the version before this one made."""
    from built_fakes import rowop, scale
    from lower_cases import gelu_exact, layernorm

    spec = spec_of(plain)
    assert spec.schema == 2 and not spec.uses_strided_schema
    assert CompileSpec.from_dict(spec.to_dict()).schema == 2
    assert declaration_digest(scale) == "ca1676d848f937753044539e65bfc0023338ab45c010230e57c2df4c9b0f84dd"
    assert declaration_digest(rowop) == "c1f6391b81b9620234d6a74a626262c524eb43e8588133dddbdfd3f42881bca3"
    assert declaration_digest(tilefish.lower(layernorm).extern) == "529934a610d6a4319c98d347ce400947709e061a0914219e0398c0aa75ffe25c"
    assert declaration_digest(tilefish.lower(gelu_exact).extern) == "1d945a997ff5a0c99a21bfc0d33989591380c3efca3dc332d7c49fe6963a3090"
    assert compile_specs(scale)[0]._fingerprint() == "3ac79383b989a2770e158930ba8d0db22682f5aee8d5630354b4ddbac627863c"


def test_a_declaration_digest_covers_the_layout():
    assert declaration_digest(head_copy) != declaration_digest(head_copy16) != declaration_digest(head_copy64)
    ok = declaration_digest(head_copy)
    assert ok == declaration_digest(head_copy)

    @tilefish.extern(triton="kernels:head_copy", num_warps=1, grid=lambda S: (S, 8), name="head_copy")
    def swapped(  # the same op with its stride parameters in the other order
        x: Strided[Tensor[S, 8, 64, bf16]], y: Out[Tensor[S, 8, 64, bf16]], x_s1: StrideOf("x", 1), x_s0: StrideOf("x", 0), D: const = 64
    ): ...

    assert declaration_digest(swapped) != ok


def test_a_spec_with_strides_round_trips_and_verifies():
    for op in (head_copy, head_copy16, head_copy64):
        spec = spec_of(op)
        again = CompileSpec.from_json(spec.to_json())
        assert again == spec and again.to_dict() == spec.to_dict() and again.schema == 3
        again.verify_against(op)
        assert again.verified and again.args[0].strides == spec.args[0].strides and again.args[2].stride_of == ("x", 0)


def test_a_spec_that_says_a_stride_is_another_one_does_not_verify():
    doc = spec_of(head_copy).to_dict()
    doc["args"][2]["stride_of"]["axis"] = 1
    with pytest.raises(TilefishSpecError, match="carries 'x_s0' as the stride of axis 1 of 'x'"):
        CompileSpec.from_dict(doc)


# -- strict reading of the new fields ---------------------------------------------------------------------------------------------------------


def doc_of(op=head_copy):
    return copy.deepcopy(spec_of(op).to_dict())


def rejects(change, message, op=head_copy):
    doc = doc_of(op)
    change(doc)
    with pytest.raises(TilefishSpecError, match=message):
        CompileSpec.from_dict(doc)


def test_the_new_fields_are_schema_3_only():
    rejects(lambda d: d.update(schema=2), "unknown fields|must have exactly the fields")
    rejects(lambda d: d["preconditions"].append({"kind": "max_extent", "arg": "x", "index": 0, "value": 2**31 - 1}), "max_extent precondition needs schema 3", op=plain)
    rejects(lambda d: d["args"][0].update(strides=[1, 1], offset="o"), "must have exactly the fields", op=plain)  # schema 2 has no strides
    rejects(lambda d: d["args"][1].update(stride_of={"arg": "x", "axis": 0}), "must have exactly the fields", op=plain)
    rejects(lambda d: d["args"][0].pop("offset"), "strides and offset come together")
    rejects(lambda d: d["args"][0].pop("strides"), "strides and offset come together")


def test_strides_and_offset_are_checked():
    rejects(lambda d: d["args"][0].update(strides=["x_s0", "x_s1"]), "strides must be a list with one entry per axis")
    rejects(lambda d: d["args"][0].update(strides=[]), "strides must be a list with one entry per axis")
    rejects(lambda d: d["args"][0].update(strides=["x_s0", -3, 1]), "must not be negative")
    rejects(lambda d: d["args"][0].update(strides=["x_s0", 1.5, 1]), "must be an int or a dimension expression")
    rejects(lambda d: d["args"][0].update(strides=["x_s0", "__import__('os')", 1]), "dimension grammar")
    rejects(lambda d: d["args"][0].update(offset="x_offset; drop()"), "dimension grammar")
    rejects(lambda d: d["args"][0].update(offset=None), "must be an int or a dimension expression")
    rejects(lambda d: d["args"][0].update(strides=["x_s0", "x_s1", 2]), "dimension|stride")  # an int stride has no scalar and is fine only if it is the declared one
    # a stride that no scalar carries
    rejects(lambda d: d["args"][0].update(strides=["x_s0", "other", 1]), "carried by no scalar|carries 'x_s1'")


def test_stride_of_is_checked():
    rejects(lambda d: d["args"][2].update(stride_of={"arg": "y", "axis": 0}), "which is not a strided pointer argument")
    rejects(lambda d: d["args"][2].update(stride_of={"arg": "x", "axis": 7}), "which has 3 axes")
    rejects(lambda d: d["args"][2].update(stride_of={"arg": "x"}), "must have exactly the fields|is missing")
    rejects(lambda d: d["args"][2].update(stride_of={"arg": "x", "axis": -1}), "must be an int >= 0")
    rejects(lambda d: d["args"][3].update(stride_of={"arg": "x", "axis": 0}), "has two stride scalars|carries 'x_s1'")
    rejects(lambda d: d["args"][2].update(count_of="x"), "the element count of a pointer or the stride of one, not both|count_of")
    rejects(lambda d: d["args"][0].update(stride_of={"arg": "x", "axis": 0}), "must have exactly the fields")


def test_the_alignment_of_a_strided_pointer_is_16_bytes_or_its_element():
    p = next(i for i, p in enumerate(doc_of()["preconditions"]) if p["kind"] == "pointer_alignment" and p["arg"] == "x")
    rejects(lambda d: d["preconditions"][p].update(bytes=4), "must be 16 bytes or its element size 2")
    rejects(lambda d: d["preconditions"][p].update(bytes=16), "attrs must be at exactly")  # promised 16 bytes, but no attribute says so to Triton
    rejects(lambda d: d["preconditions"][next(i for i, q in enumerate(d["preconditions"]) if q["arg"] == "y" and q["kind"] == "pointer_alignment")].update(bytes=2), "pointer_alignment must be 16 bytes")


def test_max_extent_and_max_elements_are_checked_on_the_right_pointers():
    rejects(lambda d: d["preconditions"].append({"kind": "max_elements", "arg": "x", "index": 0, "value": 2**31 - 1, "shape": ["S", "8", "64"]}), "max_elements must be once, on a dense pointer")
    rejects(lambda d: d["preconditions"].append({"kind": "max_extent", "arg": "y", "index": 1, "value": 2**31 - 1}), "max_extent must be once, on a strided pointer")
    rejects(lambda d: d["preconditions"].append({"kind": "max_extent", "arg": "x", "index": 0, "value": 2**31 - 1}), "max_extent must be once")
    rejects(lambda d: next(p for p in d["preconditions"] if p["kind"] == "max_extent").update(value=1000), "max_extent must be once, on a strided pointer argument, with the value 2147483647")
    rejects(lambda d: next(p for p in d["preconditions"] if p["kind"] == "max_extent").update(shape=["S"]), "must have exactly the fields|unknown")


def test_a_64_bit_scalar_is_schema_3_only_and_matches_its_dtype():
    rejects(lambda d: d["args"][2].update(type="i64"), "a runtime scalar must be i32 or i64, got 'i64'/'i32'")
    rejects(lambda d: (d["args"][2].update(type="i64"), d["signature"].update({"x_s0": "i64"})), "must be i32 or i64, got 'i64'/'i32'")
    rejects(lambda d: d["args"][2].update(type="f32", dtype="f32"), "a runtime scalar must be i32 or i64")


# -- the launch checks, with fake tensors --------------------------------------------------------------------------------------------------------


class FakeView(FakeTensor):
    """A fake tensor with strides and a storage offset."""

    def __init__(self, shape, dtype, ptr, strides):
        super().__init__(shape, dtype, ptr)
        self._strides = tuple(strides)

    def stride(self):
        return self._strides

    def is_contiguous(self):
        acc = 1
        for n, s in zip(reversed(self.shape), reversed(self._strides)):
            if n != 1 and s != acc:
                return False
            acc *= n
        return True


SLAB = 1 << 40


def run_prepare(op, view):
    prepared, error = prepare(op, [view], next_slab=SLAB)
    return prepared, error


def test_a_launch_binds_the_strides_of_the_view_and_passes_its_pointer():
    view = FakeView((40, 8, 64), "bf16", 0x7F0000000400, (1536, 64, 1))
    prepared, error = run_prepare(head_copy, view)
    assert error is None and prepared.scalars == {"x_s0": 1536, "x_s1": 64}
    assert prepared.tensors["x"].data_ptr() == 0x7F0000000400 and prepared.grid == (40, 8)


def test_a_view_that_is_not_16_byte_aligned_is_fine_for_strided_and_refused_for_strided_16():
    view = FakeView((40, 8, 64), "bf16", 0x7F0000000002, (512, 64, 1))
    assert run_prepare(head_copy, view)[1] is None
    _, error = run_prepare(head_copy16, view)
    assert isinstance(error, TilefishPreconditionError) and error.kind == "pointer_alignment" and "not aligned to 16 bytes" in str(error)


def test_a_stride_scalar_that_is_not_a_multiple_of_what_its_declaration_says_is_refused():
    view = FakeView((40, 8, 64), "bf16", 0x7F0000000000, (520, 64, 1))  # 520 is a multiple of 8, not of 16
    _, error = run_prepare(head_copy16, view)
    assert isinstance(error, TilefishPreconditionError) and error.kind == "divisible" and "x_s0" in str(error)


def test_the_inner_stride_is_fixed_and_a_negative_stride_is_refused():
    bad = FakeView((40, 8, 64), "bf16", 0x7F0000000000, (1024, 128, 2))
    _, error = run_prepare(head_copy, bad)
    assert error.kind == "stride" and "the stride of axis 2 is 2, but the signature requires 1" in str(error)
    neg = FakeView((40, 8, 64), "bf16", 0x7F0000000000, (512, -64, 1))
    _, error = run_prepare(head_copy, neg)
    assert error.kind == "stride" and "negative stride" in str(error)


def test_the_stride_of_a_size_one_axis_is_canonical_and_zero_is_a_stride():
    assert canonical_strides((1, 8, 64), (123456789, 64, 1)) == (512, 64, 1)
    assert canonical_strides((4, 1, 64), (64, 999, 1)) == (64, 64, 1)
    assert canonical_strides((4, 8, 1), (8, 1, 77)) == (8, 1, 1)
    assert canonical_strides((3, 4), None) == (4, 1)
    prepared, error = run_prepare(head_copy, FakeView((40, 8, 64), "bf16", 0x7F0000000000, (0, 64, 1)))  # an expanded view
    assert error is None and prepared.scalars["x_s0"] == 0
    one, error = run_prepare(head_copy, FakeView((1, 8, 64), "bf16", 0x7F0000000000, (123456789123, 64, 1)))  # whatever torch says for a size-1 axis
    assert error is None and one.scalars["x_s0"] == 512


def test_a_stride_that_does_not_fit_32_bits_is_refused_unless_the_kernel_takes_64():
    huge = FakeView((40, 8, 64), "bf16", 0x7F0000000000, (2**31, 64, 1))
    _, error = run_prepare(head_copy, huge)
    assert error is not None and error.kind in ("shape", "max_extent")
    prepared, error = run_prepare(head_copy64, huge)
    assert error is None and prepared.scalars["x_s0"] == 2**31


def test_the_extent_of_a_view_is_bounded_not_its_element_count():
    assert extent_elements((40, 8, 64), (1536, 64, 1)) == 39 * 1536 + 7 * 64 + 63 + 1
    assert extent_elements((40, 8, 64), (0, 64, 1)) == 7 * 64 + 63 + 1
    assert extent_elements((0, 8), (8, 1)) == 0
    view = FakeView((40, 8, 64), "bf16", 0x7F0000000000, (2**26, 64, 1))  # 40 * 8 * 64 elements, but they span 2**26 * 39 + ...
    _, error = run_prepare(head_copy, view)
    assert error is not None and error.kind == "max_extent" and "spans" in str(error)
    _, error = run_prepare(head_copy64, view)
    assert error is None


def test_a_view_that_overlaps_an_output_is_an_alias():
    # the output of a launch is a new allocation, so a view never overlaps it; an InOut view would: not allowed in this version
    with pytest.raises(TilefishDeclarationError, match="only input tensors can be strided"):

        @tilefish.extern(triton="m:f", grid=(1,))
        def f(x: tilefish.InOut[Strided[Tensor[4, 8, bf16]]], a: StrideOf("x", 0)): ...


def test_a_device_scalar_is_a_pointer_with_no_axes_in_the_spec():
    @tilefish.extern(triton="kernels:masked", grid=lambda S: (S,))
    def masked(x: Tensor[S, 128, f32], y: Out[Tensor[S, 128, f32]], n_tok: DevScalar[i32], D: const = 128): ...

    spec = spec_of(masked)
    n = spec.args[2]
    assert (n.kind, n.type, n.dtype, n.shape, n.shape_dims, n.strides) == ("pointer", "*i32", "i32", (), (), None)
    assert [p.to_dict() for p in spec.preconditions if p.arg == "n_tok"] == [{"kind": "pointer_alignment", "arg": "n_tok", "index": 2, "bytes": 16}]
    assert spec.attrs[(2,)] == [["tt.divisibility", 16]] and abi_of(spec)["params"][2]["shape"] == []
    assert spec.schema == 2  # a pointer with an empty shape needs nothing new
    assert CompileSpec.from_json(spec.to_json()) == spec

    @tilefish.extern(triton="kernels:masked", grid=lambda S: (S,), name="masked")
    def as_tensor(x: Tensor[S, 128, f32], y: Out[Tensor[S, 128, f32]], n_tok: Tensor[i32], D: const = 128): ...

    # the same ABI, but another declaration: a rank-0 tensor is not a device scalar (it is not read with tilefish.load)
    assert abi_of(spec_of(as_tensor))["params"][2] == abi_of(spec)["params"][2] and declaration_digest(as_tensor) != declaration_digest(masked)
