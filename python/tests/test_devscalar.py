"""``DevScalar[dtype]``: one element in device memory, read by a traced body with ``tilefish.load(s)`` and by a lowered kernel with one ``tl.load(ptr)``.

In an ABI it is a pointer with the shape ``[]``; at the API it is a tensor of one element. It is not an operand of any operation until it is loaded,
and a lowered kernel (the row and the elementwise template) still takes contiguous tensors only: a view is refused with the rule that says so. What
runs on a GPU is ``test_gpu_devscalar.py``.
"""

import pytest

import tilefish
from tilefish import DevScalar, Dim, Out, Tensor, TilefishLoweringError, TilefishTypeError, bf16, f32, i32
from tilefish.aot import compile_specs

M = Dim("M")
T = Dim("T", divisible_by=16)
D = Dim("D", 256)


@tilefish.kernel
def scale_rows(x: Tensor[M, D, f32], s: DevScalar[f32]) -> Tensor[M, D, f32]:
    return x * tilefish.load(s)


@tilefish.kernel
def scale_flat(x: Tensor[T, f32], s: DevScalar[f32], b: DevScalar[f32]) -> Tensor[T, f32]:
    return x * tilefish.load(s) + tilefish.load(b)


@tilefish.kernel
def scalar_first(s: DevScalar[f32], x: Tensor[T, f32]) -> Tensor[T, f32]:
    k = tilefish.load(s) + 1.0  # a scalar computed from a scalar
    return x * k


# -- the traced body --------------------------------------------------------------------------------------------------------------------------


def test_load_is_a_node_and_its_result_a_rank_0_tensor():
    g = scale_rows.graph
    assert [n.op for n in g.nodes] == ["load", "mul"]
    assert g.nodes[0].outputs[0].type == Tensor[f32] and g.nodes[0].inputs[0] is g.inputs[1]
    assert g.inputs[1].type.devscalar and g.inputs[1].type.shape == () and g.outputs[0].type == Tensor[M, D, f32]


def test_a_devscalar_is_a_graph_input_and_is_passed_through_a_call():
    @tilefish.extern(triton="m:masked", grid=lambda M: (M,))
    def masked(x: Tensor[M, 8, f32], y: Out[Tensor[M, 8, f32]], n: DevScalar[i32]): ...

    @tilefish.graph
    def g(x: Tensor[M, 8, f32], n: DevScalar[i32]):
        return masked(x, n)

    assert g.graph.outputs[0].type == Tensor[M, 8, f32]
    with pytest.raises(TilefishTypeError, match=r"masked: argument 'n': expected DevScalar\[i32\], got Tensor\[i32\]"):

        @tilefish.graph
        def not_a_devscalar(x: Tensor[M, 8, f32], n: Tensor[i32]):
            return masked(x, n)

    with pytest.raises(TilefishTypeError, match=r"argument 'x': expected Tensor\[M, 8, f32\], got DevScalar\[f32\]"):

        @tilefish.graph
        def swapped(x: Tensor[M, 8, f32], s: DevScalar[f32], n: DevScalar[i32]):
            return masked(s, n)

    with pytest.raises(TilefishTypeError, match="expected dtype i32, got f32"):

        @tilefish.graph
        def wrong_dtype(x: Tensor[M, 8, f32], s: DevScalar[f32]):
            return masked(x, s)


def test_a_loaded_scalar_is_an_ordinary_rank_0_tensor_that_broadcasts():
    g = scalar_first.graph
    assert [n.op for n in g.nodes] == ["load", "add", "mul"] and g.outputs[0].type == Tensor[T, f32]


# -- the lowered kernel -----------------------------------------------------------------------------------------------------------------------


def test_the_row_template_reads_a_devscalar_with_one_load_and_no_mask():
    low = scale_rows.lower(template="row")
    assert low.template == "row" and scale_rows.lower().template == "elementwise"  # (the elementwise template is tried first)
    assert "    v0 = tl.load(x + offs, mask=mask, other=0)  # x" in low.source
    assert "    v1 = tl.load(s)\n" in low.source and "v2 = v0 * v1" in low.source
    assert "tl.load(s +" not in low.source and "s: " not in low.source.split("def ")[1].split("):")[0]  # the pointer is a plain parameter
    ext = low.extern
    s = ext.signature.params[1]
    assert (s.name, s.kind) == ("s", "tensor") and s.type.devscalar and s.type.dtype is f32
    assert [p.name for p in ext.signature.params] == ["x", "s", "out", "D", "BLOCK_D"]


def test_the_elementwise_template_takes_scalars_that_are_not_part_of_the_shape():
    low = scale_flat.lower()
    assert low.template == "elementwise"
    assert "= tl.load(s)\n" in low.source and "= tl.load(b)\n" in low.source
    assert [p.name for p in low.extern.signature.params] == ["x", "s", "b", "out", "n", "BLOCK"]
    assert low.extern.element_counts == {"n": "x"}  # the element count is that of x, never of the scalar that comes first in a signature
    first = scalar_first.lower()
    assert first.extern.element_counts == {"n": "x"}


def test_the_spec_of_a_lowered_devscalar_kernel_has_a_pointer_with_no_axes():
    for kernel, name in ((scale_rows, "s"), (scale_flat, "s"), (scale_flat, "b")):
        for spec in compile_specs(kernel.lower().extern):
            a = next(a for a in spec.args if a.name == name)
            assert (a.kind, a.shape, a.shape_dims, a.strides) == ("pointer", (), (), None) and a.type == "*fp32"
            assert not [p for p in spec.preconditions if p.arg == name and p.kind != "pointer_alignment"]
            assert spec.attrs[(a.index,)] == [["tt.divisibility", 16]]
            assert spec.schema == 2  # a pointer with an empty shape is not new


def test_a_lowered_kernel_with_a_devscalar_is_deterministic():
    assert scale_rows.lower().source == tilefish.lower(scale_rows).source
    assert tilefish.lower(scale_rows).source_sha256 == scale_rows.lower().source_sha256


def test_the_lowering_names_the_rule_that_a_devscalar_cannot_be_the_result_of_a_kernel():
    @tilefish.kernel
    def bad(x: Tensor[M, D, f32], s: DevScalar[f32]) -> Tensor[f32]:
        return tilefish.load(s)

    with pytest.raises(TilefishLoweringError, match="scalar"):
        bad.lower()

    @tilefish.kernel
    def unranked(x: Tensor[M, D, f32], s: Tensor[f32]) -> Tensor[M, D, f32]:
        return x * s

    with pytest.raises(TilefishLoweringError, match="rank 0"):
        unranked.lower()


# -- views are not lowered ---------------------------------------------------------------------------------------------------------------------


def test_a_lowered_kernel_refuses_every_view_with_the_rule_that_says_so():
    @tilefish.kernel
    def reshaped(x: Tensor[M, 256, f32]) -> Tensor[M, 4, 64, f32]:
        return x.reshape(M, 4, 64) + 0.0

    @tilefish.kernel
    def permuted(x: Tensor[M, 4, 64, f32]) -> Tensor[4, M, 64, f32]:
        return x.permute(1, 0, 2) + 0.0

    @tilefish.kernel
    def selected(x: Tensor[M, 4, 64, f32]) -> Tensor[M, 64, f32]:
        return x[:, 1] + 0.0

    @tilefish.kernel
    def split_it(x: Tensor[M, 128, f32]) -> Tensor[M, 64, f32]:
        a, b = x.split(64, axis=1)
        return a + b

    for kernel, op in ((reshaped, "reshape"), (permuted, "permute"), (selected, "select")):
        with pytest.raises(TilefishLoweringError, match=rf"rule 'view' failed: .*{op}.* is a view operation, and strided views are not lowered") as e:
            kernel.lower()
        assert e.value.rule == "view"
    with pytest.raises(TilefishLoweringError, match="rule 'split' failed: split is not lowered.*a strided view goes to an extern with a Strided") as e:
        split_it.lower()
    assert e.value.rule == "split"


def test_a_view_parameter_of_a_kernel_is_not_a_declarable_annotation_and_the_old_rule_stands():
    with pytest.raises(tilefish.TilefishDeclarationError, match="only an extern op can take"):

        @tilefish.kernel
        def k(x: tilefish.Strided[Tensor[4, 8, bf16]]) -> Tensor[4, 8, bf16]:
            return x
