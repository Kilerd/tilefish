"""Declaration forms: decorators with and without arguments, annotations, solved dimensions."""

import pytest

import tilefish
from tilefish import Dim, Tensor, TilefishTypeError, bf16, f32

M = Dim("M")


def test_tuple_return_annotations_and_decorator_forms():
    @tilefish.kernel
    def pair(x: Tensor[M, 4, f32]) -> tuple[Tensor[M, 4, f32], Tensor[M, 2, f32]]:
        a, _ = x.split(2, axis=1)
        return x, a

    assert pair.signature.returns == (Tensor[M, 4, f32], Tensor[M, 2, f32])

    @tilefish.library
    def bare(x: Tensor[M, 4, f32]) -> Tensor[M, 4, f32]: ...

    assert bare.name == "bare" and bare.acc is None

    @tilefish.graph(name="renamed")
    def g(x: Tensor[M, 4, f32]):
        a, b = pair(x)
        return bare(a), b

    assert g.name == "renamed" and g.graph.name == "renamed"
    assert [str(v.type) for v in g.graph.outputs] == ["Tensor[M, 4, f32]", "Tensor[M, 2, f32]"]


def test_string_annotations_are_evaluated_in_the_function_globals():
    ns = {"Tensor": Tensor, "bf16": bf16, "M": M, "tilefish": tilefish}
    exec(
        "@tilefish.kernel\n"
        "def ident(x: 'Tensor[M, 4, bf16]') -> 'Tensor[M, 4, bf16]':\n"
        "    return x\n",
        ns,
    )
    assert ns["ident"].signature.returns == (Tensor[M, 4, bf16],)


def test_solved_dimensions_must_be_positive():
    P = Dim("P")

    @tilefish.library("shifted")
    def shifted(x: Tensor[P + 5, bf16]) -> Tensor[P, bf16]: ...

    with pytest.raises(
        TilefishTypeError,
        match=r"shifted: argument 'x': dimension 0 is 3, which cannot match P \+ 5 \(would need P = -2\)",
    ):

        @tilefish.graph
        def g(x: Tensor[3, bf16]):
            return shifted(x)

    @tilefish.graph
    def ok(x: Tensor[8, bf16]):
        return shifted(x)

    assert ok.graph.outputs[0].type == Tensor[3, bf16]


def test_ops_have_readable_text():
    @tilefish.library("scale", acc=f32)
    def scale(x: Tensor[M, 4, bf16], *, factor: tilefish.const[f32] = 2.0) -> Tensor[M, 4, bf16]: ...

    assert str(scale) == "library scale(x: Tensor[M, 4, bf16], factor: const[f32] = 2.0) -> Tensor[M, 4, bf16]"
