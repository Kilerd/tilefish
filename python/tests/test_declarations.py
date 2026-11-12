"""Declaration forms: decorators with and without arguments, annotations, solved dimensions."""

import types

import pytest

import tilefish
from tilefish import Dim, Out, Tensor, TilefishTypeError, bf16, const, f32

M = Dim("M")


def test_tuple_return_annotations_and_decorator_forms():
    @tilefish.kernel
    def pair(x: Tensor[M, 4, f32]) -> tuple[Tensor[M, 4, f32], Tensor[M, 2, f32]]:
        a, _ = x.split(2, axis=1)
        return x, a * 1  # a split along axis 1 is a view: compute from it

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


# -- decorator misuse ----------------------------------------------------------------------------------------


def test_extern_without_arguments_is_a_declaration_error_that_says_how_to_call_it():
    def stub(x: Tensor[M, bf16]): ...

    with pytest.raises(tilefish.TilefishDeclarationError, match=r"@tilefish\.extern takes arguments.*triton=\"module:function\""):
        tilefish.extern(stub)  # what a bare @tilefish.extern does
    with pytest.raises(tilefish.TilefishDeclarationError, match="takes arguments"):
        tilefish.extern()
    with pytest.raises(tilefish.TilefishDeclarationError, match="takes arguments"):

        @tilefish.extern
        def bare(x: Tensor[M, bf16]): ...


def test_graph_and_kernel_decorators_check_what_they_decorate():
    with pytest.raises(tilefish.TilefishDeclarationError, match=r"@tilefish\.graph decorates a function"):
        tilefish.graph("name")
    with pytest.raises(tilefish.TilefishDeclarationError, match=r"@tilefish\.kernel decorates a function"):
        tilefish.kernel(3)
    with pytest.raises(tilefish.TilefishDeclarationError, match="the name must be a string"):
        tilefish.library(3)


# -- the stub against the Triton kernel object -----------------------------------------------------------------


class FakeKernel:
    """Duck-types what a triton.JITFunction exposes: .fn, .arg_names and .params[i].is_constexpr."""

    def __init__(self, module, name, arg_names, constexpr=None):
        def fn():  # the wrapped python function
            pass

        fn.__module__, fn.__name__ = module, name
        self.fn = fn
        self.arg_names = list(arg_names)
        if constexpr is not None:
            self.params = [types.SimpleNamespace(name=n, is_constexpr=c) for n, c in zip(arg_names, constexpr)]


def test_the_wrapped_function_names_the_kernel():
    k = FakeKernel("my_kernels", "gelu_kernel", ["X", "Y"])
    assert type(k).__module__ != "my_kernels"  # the wrapper object itself would give the wrong module

    @tilefish.extern(triton=k, grid=(1,))
    def e(X: Tensor[M, bf16], Y: Out[Tensor[M, bf16]]): ...

    assert e.triton == "my_kernels:gelu_kernel"


def test_the_stub_is_checked_against_the_kernel_parameters():
    k = FakeKernel("my_kernels", "norm", ["X", "W", "Y", "N", "BLOCK"], constexpr=[False, False, False, True, True])

    @tilefish.extern(triton=k, grid=lambda M: (M,))
    def ok(X: Tensor[M, 64, bf16], W: Tensor[64, bf16], Y: Out[Tensor[M, 64, bf16]], N: const = 64, BLOCK: const = 128): ...

    assert ok.triton == "my_kernels:norm"

    # same names, wrong order
    with pytest.raises(
        tilefish.TilefishDeclarationError,
        match=r"extern op 'swapped': the stub's parameters \['W', 'X', 'Y', 'N', 'BLOCK'\] do not match the parameters of the "
        r"Triton kernel 'my_kernels:norm' \['X', 'W', 'Y', 'N', 'BLOCK'\] \(first difference at position 0\)",
    ):

        @tilefish.extern(triton=k, grid=lambda M: (M,))
        def swapped(W: Tensor[64, bf16], X: Tensor[M, 64, bf16], Y: Out[Tensor[M, 64, bf16]], N: const = 64, BLOCK: const = 128): ...

    # a missing trailing parameter
    with pytest.raises(tilefish.TilefishDeclarationError, match=r"first difference at position 4"):

        @tilefish.extern(triton=k, grid=lambda M: (M,))
        def short(X: Tensor[M, 64, bf16], W: Tensor[64, bf16], Y: Out[Tensor[M, 64, bf16]], N: const = 64): ...

    # a parameter the kernel does not have
    with pytest.raises(tilefish.TilefishDeclarationError, match=r"first difference at position 5"):

        @tilefish.extern(triton=k, grid=lambda M: (M,))
        def long(X: Tensor[M, 64, bf16], W: Tensor[64, bf16], Y: Out[Tensor[M, 64, bf16]], N: const = 64, BLOCK: const = 128, Z: const = 1): ...

    # a different name
    with pytest.raises(tilefish.TilefishDeclarationError, match=r"first difference at position 3"):

        @tilefish.extern(triton=k, grid=lambda M: (M,))
        def renamed(X: Tensor[M, 64, bf16], W: Tensor[64, bf16], Y: Out[Tensor[M, 64, bf16]], N_COLS: const = 64, BLOCK: const = 128): ...


def test_const_in_the_stub_must_be_constexpr_in_the_kernel_and_back():
    k = FakeKernel("my_kernels", "k", ["X", "n", "BLOCK"], constexpr=[False, False, True])
    nn = Dim("nn")

    @tilefish.extern(triton=k, grid=(1,))
    def ok(X: Tensor[nn, bf16], n: nn, BLOCK: const = 64): ...

    with pytest.raises(
        tilefish.TilefishDeclarationError,
        match=r"parameter 'n' is a const in the stub but not a constexpr in the Triton kernel 'my_kernels:k'",
    ):

        @tilefish.extern(triton=k, grid=(1,))
        def runtime_in_kernel(X: Tensor[nn, bf16], n: const = 4, BLOCK: const = 64): ...

    with pytest.raises(
        tilefish.TilefishDeclarationError,
        match=r"parameter 'BLOCK' is not a const in the stub but a constexpr in the Triton kernel",
    ):

        @tilefish.extern(triton=k, grid=(1,))
        def runtime_in_stub(X: Tensor[nn, bf16], n: nn, BLOCK: nn): ...


def test_without_a_parameter_list_nothing_is_checked():
    plain = FakeKernel("my_kernels", "k", ["X"])
    del plain.arg_names

    @tilefish.extern(triton=plain, grid=(1,))
    def anything(a: Tensor[M, bf16], b: const = 1): ...

    assert anything.triton == "my_kernels:k"

    names_only = FakeKernel("my_kernels", "k", ["X", "n"])  # no params list: constexpr is not checked

    @tilefish.extern(triton=names_only, grid=(1,))
    def only_names(X: Tensor[M, bf16], n: const = 3): ...

    assert only_names.name == "only_names"


def test_an_autotuner_like_wrapper_reaches_the_parameter_list_through_fn():
    jit = FakeKernel("my_kernels", "k", ["X", "BLOCK"], constexpr=[False, True])

    class Autotuner:  # triton.runtime.Autotuner keeps the JIT function in .fn
        def __init__(self, fn):
            self.fn = fn

    tuned = Autotuner(jit)
    ok_names = ["X", "BLOCK"]

    @tilefish.extern(triton=tuned, grid=(1,))
    def ok(X: Tensor[M, bf16], BLOCK: const = 64): ...

    assert list(jit.arg_names) == ok_names
    # the Autotuner's fn is the JIT object, which has no __module__ of its own beyond its class: the kernel is
    # found through the chain only if one level has a module and a name
    with pytest.raises(tilefish.TilefishDeclarationError, match="module:function"):
        tilefish.extern(triton=Autotuner(types.SimpleNamespace(arg_names=["X"])), grid=(1,))

    with pytest.raises(tilefish.TilefishDeclarationError, match="do not match the parameters"):

        @tilefish.extern(triton=tuned, grid=(1,))
        def wrong(Y: Tensor[M, bf16], BLOCK: const = 64): ...


def test_the_documented_star_pattern_keeps_declaration_order():
    k = FakeKernel("my_kernels", "k", ["x", "BLOCK", "y", "n"], constexpr=[False, True, False, False])
    nn = Dim("nn")

    @tilefish.extern(triton=k, grid=(1,))
    def e(x: Tensor[nn, bf16], *, BLOCK: const = 64, y: Out[Tensor[nn, bf16]], n: nn): ...

    assert [p.name for p in e.signature.params] == ["x", "BLOCK", "y", "n"]
    assert "bare ``*``" in tilefish.extern.__doc__ and "keyword" in tilefish.extern.__doc__


# -- wrappers: the parameter list comes from wherever it exists in the .fn chain ------------------------------------------------------


class Wrapper:
    """An autotuner or heuristics wrapper: it may expose arg_names but not params, or neither."""

    def __init__(self, fn, arg_names=None, params=None):
        self.fn = fn
        if arg_names is not None:
            self.arg_names = list(arg_names)
        if params is not None:
            self.params = params


def _params(names, constexpr):
    return [types.SimpleNamespace(name=n, is_constexpr=c) for n, c in zip(names, constexpr)]


NAMES = ["X", "n", "BLOCK"]


def test_an_autotuner_with_arg_names_but_no_params_still_gets_the_constexpr_check_from_the_jit_below_it():
    jit = FakeKernel("my_kernels", "k", NAMES, constexpr=[False, False, True])
    tuned = Wrapper(jit, arg_names=NAMES)  # arg_names but no params: the first level that has arg_names
    nn = Dim("nn")

    @tilefish.extern(triton=tuned, grid=(1,))
    def ok(X: Tensor[nn, bf16], n: nn, BLOCK: const = 64): ...

    assert ok.triton == "my_kernels:k"
    with pytest.raises(
        tilefish.TilefishDeclarationError,
        match=r"parameter 'BLOCK' is not a const in the stub but a constexpr in the Triton kernel",
    ):

        @tilefish.extern(triton=tuned, grid=(1,))
        def stub_runtime(X: Tensor[nn, bf16], n: nn, BLOCK: nn): ...

    with pytest.raises(
        tilefish.TilefishDeclarationError,
        match=r"parameter 'n' is a const in the stub but not a constexpr in the Triton kernel",
    ):

        @tilefish.extern(triton=tuned, grid=(1,))
        def stub_const(X: Tensor[nn, bf16], n: const = 4, BLOCK: const = 64): ...


def test_a_heuristics_wrapper_over_an_autotuner_over_a_jit_is_checked_too():
    jit = FakeKernel("my_kernels", "k", NAMES, constexpr=[False, False, True])
    chain = Wrapper(Wrapper(jit, arg_names=NAMES), arg_names=NAMES)  # heuristics(autotuner(jit))
    nn = Dim("nn")

    @tilefish.extern(triton=chain, grid=(1,))
    def ok(X: Tensor[nn, bf16], n: nn, BLOCK: const = 64): ...

    with pytest.raises(tilefish.TilefishDeclarationError, match="not a const in the stub but a constexpr"):

        @tilefish.extern(triton=chain, grid=(1,))
        def wrong(X: Tensor[nn, bf16], n: nn, BLOCK: nn): ...

    # params only, found on a wrapper that has no arg_names: names come from the params
    only_params = Wrapper(types.SimpleNamespace(__module__="my_kernels", __name__="k"), params=_params(NAMES, [False, False, True]))

    @tilefish.extern(triton=only_params, grid=(1,))
    def ok2(X: Tensor[nn, bf16], n: nn, BLOCK: const = 64): ...

    with pytest.raises(tilefish.TilefishDeclarationError, match="first difference at position 0"):

        @tilefish.extern(triton=only_params, grid=(1,))
        def renamed(Y: Tensor[nn, bf16], n: nn, BLOCK: const = 64): ...


def test_wrappers_that_disagree_about_the_kernel_are_rejected():
    jit = FakeKernel("my_kernels", "k", NAMES, constexpr=[False, False, True])

    other_names = Wrapper(jit, arg_names=["X", "m", "BLOCK"])
    with pytest.raises(tilefish.TilefishDeclarationError, match=r"disagree about its parameter names"):
        tilefish.extern(triton=other_names, grid=(1,))

    other_flags = Wrapper(
        FakeKernel("my_kernels", "k", NAMES, constexpr=[False, False, True]),
        arg_names=NAMES,
        params=_params(NAMES, [False, True, True]),
    )
    with pytest.raises(tilefish.TilefishDeclarationError, match=r"disagree about its constexpr flags"):
        tilefish.extern(triton=other_flags, grid=(1,))

    params_vs_names = Wrapper(jit, params=_params(["X", "n", "OTHER"], [False, False, True]))
    with pytest.raises(tilefish.TilefishDeclarationError, match=r"disagree about its parameter names"):
        tilefish.extern(triton=params_vs_names, grid=(1,))

    short_flags = types.SimpleNamespace(
        __module__="my_kernels", __name__="k", arg_names=NAMES, params=_params(NAMES[:2], [False, False])
    )
    with pytest.raises(tilefish.TilefishDeclarationError, match="3 parameter names but 2 constexpr flags|disagree"):
        tilefish.extern(triton=short_flags, grid=(1,))

    broken = types.SimpleNamespace(__module__="my_kernels", __name__="k", arg_names=5)
    with pytest.raises(tilefish.TilefishDeclarationError, match="not a list"):
        tilefish.extern(triton=broken, grid=(1,))
