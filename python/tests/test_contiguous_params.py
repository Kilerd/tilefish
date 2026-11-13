"""Every tensor parameter that is neither ``Strided[...]`` nor a ``DevScalar`` is contiguous: refused otherwise at the declaration, and checked at every launch.

A ``TensorType`` annotation with other strides, an offset or no alignment used to be accepted as it was, and a launch then checked contiguity only when the type said
so (``if ptype.contiguous``): ``InOut[TensorType((N,), f32, strides=(0,))]`` took ``small.expand(8192)``, and the kernel wrote 8192 elements from a pointer that owns
4096 (the neighbouring allocation was overwritten on a GPU: ``test_gpu_strided.py``). A plan bound such a graph parameter as row-major whatever its type said. The
Rust host refuses the call. Now the declaration refuses the annotation (it says to use ``Strided[...]``), whatever the operator (kernel, extern, library, graph), and
the launch refuses any non-contiguous tensor for such a parameter, whatever its type.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import pytest

import tilefish
from tilefish import (
    Dim,
    InOut,
    Out,
    Strided,
    StrideOf,
    Tensor,
    TensorType,
    TilefishDeclarationError,
    TilefishPlanError,
    TilefishPreconditionError,
    bf16,
    f32,
)
from tilefish._launch import check_tensor, tensor_info
from tilefish.signature import Param

N = Dim("N")
M = Dim("M", buckets=[64])
K = Dim("K", buckets=[64])

STRIDE_0 = TensorType((N,), f32, strides=(0,))
TRANSPOSED = TensorType((N, 8), f32, strides=(1, N))
OFFSET = TensorType((N,), f32, offset=4)  # 16 bytes: aligned, dense, but not at the start of its allocation
UNALIGNED = TensorType((N,), f32, aligned=False)
OPAQUE = Tensor[N, f32].as_view()
BAD_TYPES = {"stride 0": STRIDE_0, "transposed": TRANSPOSED, "an offset": OFFSET, "not aligned": UNALIGNED, "a view of unknown layout": OPAQUE}
REFUSED = r"is not a contiguous tensor.*Strided\[Tensor\[\.\.\.\]\]"


@pytest.mark.parametrize("label", sorted(BAD_TYPES))
def test_a_kernel_parameter_that_is_not_contiguous_is_refused_at_the_declaration(label):
    t = BAD_TYPES[label]
    with pytest.raises(TilefishDeclarationError, match=rf"kernel 'k': parameter 'x' is annotated with .*{REFUSED}"):

        @tilefish.kernel(name="k")
        def k(x: t) -> Tensor[N, f32]:
            return x + 1.0

    with pytest.raises(TilefishDeclarationError, match=rf"kernel 'k': parameter 'y' .*{REFUSED}"):

        @tilefish.kernel(name="k")
        def k_inout(x: Tensor[N, f32], y: InOut[t]):
            tilefish.store(y, x)

    with pytest.raises(TilefishDeclarationError, match=rf"kernel 'k': parameter 'y' .*{REFUSED}"):

        @tilefish.kernel(name="k")
        def k_out(x: Tensor[N, f32], y: Out[t]):
            tilefish.store(y, x)


@pytest.mark.parametrize("label", sorted(BAD_TYPES))
def test_an_extern_parameter_that_is_not_contiguous_is_refused_at_the_declaration(label):
    t = BAD_TYPES[label]
    for wrap in (lambda t: t, lambda t: InOut[t], lambda t: Out[t]):
        ann = wrap(t)
        with pytest.raises(TilefishDeclarationError, match=rf"extern op 'fill': parameter 'y' .*{REFUSED}"):

            @tilefish.extern(triton="m:fill", grid=lambda N: (N,), name="fill")
            def fill(x: Tensor[N, f32], y: ann): ...

    with pytest.raises(TilefishDeclarationError, match=rf"extern op 'ret': return value 0 .*{REFUSED}"):

        @tilefish.extern(triton="m:ret", grid=lambda N: (N,), name="ret")
        def ret(x: Tensor[N, f32]) -> t: ...


def test_the_reviewers_inout_with_stride_0_is_refused_and_strided_is_how_a_view_is_declared():
    with pytest.raises(TilefishDeclarationError, match=r"masked_fill.*parameter 'x' is annotated with View\[N, f32\] \(strides \(0\), offset 0\).*Strided"):

        @tilefish.extern(triton="m:masked_fill", grid=lambda N: (N,))
        def masked_fill(x: InOut[TensorType((N,), f32, strides=(0,))]): ...

    # an input that is a view is Strided[...]: the declaration says so, and the strides are bound at a call
    @tilefish.extern(triton="m:reads_view", grid=lambda N: (N,))
    def reads_view(x: Strided[Tensor[N, f32]], y: Out[Tensor[N, f32]], x_s0: StrideOf("x", 0)): ...

    assert reads_view.signature.params[0].strided


@pytest.mark.parametrize("label", sorted(BAD_TYPES))
def test_a_library_parameter_that_is_not_contiguous_is_refused_at_the_declaration(label):
    t = BAD_TYPES[label]
    with pytest.raises(TilefishDeclarationError, match=rf"library op 'lib': parameter 'a' .*{REFUSED}"):

        @tilefish.library("lib")
        def lib(a: t) -> Tensor[N, f32]: ...

    bad = TensorType((M, K), bf16, strides=(1, M)) if label == "transposed" else TensorType((M, K), bf16, offset=8) if label == "an offset" else Tensor[M, K, bf16].as_view()
    with pytest.raises(TilefishDeclarationError, match=rf"library op 'gemm': parameter 'b' .*{REFUSED}"):

        @tilefish.library("gemm", impl="cublaslt.gemm")
        def gemm(a: Tensor[M, K, bf16], b: bad) -> Tensor[M, M, bf16]: ...


@pytest.mark.parametrize("label", sorted(BAD_TYPES))
def test_a_graph_parameter_that_is_not_contiguous_is_refused_at_the_declaration(label):
    t = BAD_TYPES[label]
    with pytest.raises(TilefishDeclarationError, match=rf"graph 'g': parameter 'x' .*{REFUSED}"):

        @tilefish.graph(name="g")
        def g(x: t):
            return x


def test_dense_annotations_and_devscalars_are_accepted():
    @tilefish.extern(triton="m:ok", grid=lambda N: (N,))
    def ok(x: Tensor[N, f32], y: InOut[TensorType((N, 8), f32, strides=(8, 1))], s: tilefish.DevScalar[tilefish.i32], z: Out[Tensor[N, f32]]): ...

    assert [p.type.contiguous for p in ok.signature.params] == [True, True, True, True]


class _Tensor:
    """What tensor_info reads off a torch tensor: a view of stride 0 is not contiguous."""

    def __init__(self, shape, strides, contiguous, ptr=1 << 40, dtype="torch.float32"):
        self.shape, self._strides, self._contiguous, self._ptr = tuple(shape), tuple(strides), contiguous, ptr
        self.dtype = type("D", (), {"__str__": lambda s: dtype})()
        self.device = "cuda:0"

    def stride(self):
        return self._strides

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self._ptr

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def element_size(self):
        return 4


@pytest.mark.parametrize("label", sorted(BAD_TYPES))
@pytest.mark.parametrize("access", ["in", "inout", "out"])
def test_a_launch_refuses_a_tensor_that_is_not_contiguous_whatever_the_type_of_its_parameter_says(label, access):
    """Past a declaration (a Param made by hand, as an internal signature could be): the launch check does not ask the type."""
    t = BAD_TYPES[label]
    expanded = _Tensor((8192,) if t.rank == 1 else (8192, 8), (0,) if t.rank == 1 else (1, 8192), contiguous=False)
    p = Param("x", 0, "tensor", access, type=t)
    with pytest.raises(TilefishPreconditionError, match="is not contiguous") as e:
        check_tensor(p, tensor_info("x", expanded))
    assert e.value.kind == "contiguous" and e.value.arg == "x"
    dense = _Tensor((8192,) if t.rank == 1 else (8192, 8), (1,) if t.rank == 1 else (8, 1), contiguous=True)
    check_tensor(p, tensor_info("x", dense))  # a contiguous tensor is what such a parameter takes


def test_a_strided_parameter_and_a_devscalar_are_not_held_to_contiguity():
    view = _Tensor((8192,), (0,), contiguous=False)
    check_tensor(Param("x", 0, "tensor", "in", type=TensorType((N,), f32, strides=(Dim("x_s0"),), offset=Dim("x_offset"), aligned=False, normalize=False), strided=True), tensor_info("x", view))
    one = _Tensor((1,), (0,), contiguous=False, dtype="torch.int32")
    check_tensor(Param("n", 0, "tensor", "in", type=tilefish.DevScalar[tilefish.i32]), tensor_info("n", one))


def test_a_plan_refuses_a_graph_parameter_that_is_not_contiguous_even_past_the_declaration(monkeypatch):
    from tilefish import signature

    monkeypatch.setattr(signature, "_require_contiguous", lambda *a: None)

    @tilefish.extern(triton="m:copy", grid=lambda N: (N,))
    def copy(x: Strided[Tensor[N, 8, f32]], y: Out[Tensor[N, 8, f32]], x_s0: StrideOf("x", 0)): ...

    B = Dim("B", buckets=[64])

    @tilefish.graph
    def g(x: TensorType((B, 8), f32, strides=(0, 1))):  # the reviewer's graph: planned as row-major, the kernel read 64 * 8 elements of a tensor of 8
        return copy(x)

    with pytest.raises(TilefishPlanError, match=r"the parameter 'x' has the type View\[B, 8, f32\] \(strides \(0, 1\), offset 0\), which is not a contiguous tensor"):
        tilefish.plan(g)
