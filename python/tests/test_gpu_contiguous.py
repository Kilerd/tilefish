"""On a GPU: a tensor parameter that is neither ``Strided[...]`` nor a ``DevScalar`` is contiguous, so a view never reaches a kernel that writes or reads it as dense.

The reviewer's repro: an extern ``fill(x: InOut[TensorType((N,), f32, strides=(0,))])`` was declared, ``small.expand(8192)`` passed the launch check (the type said
"not contiguous", so contiguity was not checked), and the kernel wrote 8192 elements from a pointer that owns one: the neighbouring memory was overwritten. A graph with
a ``strides=(0, 1)`` parameter read about 512 KB past its storage. Both annotations are refused at the declaration now, and a launch refuses a tensor that is not
contiguous whatever the type of its parameter says (``test_contiguous_params.py`` on the Mac).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

from gpu_kernels_p3 import fill_kernel  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import (  # noqa: E402
    Dim,
    InOut,
    Strided,
    StrideOf,
    Tensor,
    TensorType,
    TilefishDeclarationError,
    TilefishPreconditionError,
    const,
    f32,
    signature,
)

Count = Dim("Count")
R = Dim("R", buckets=[64])


def _fill(annotation, name):
    @tilefish.extern(triton=fill_kernel, num_warps=4, grid=lambda Count: (tilefish.cdiv(Count, 1024),), name=name)
    def fill(x: annotation, n_elements: Count, *, VALUE: const[f32] = 7.0, BLOCK: const = 1024): ...

    return fill


def test_the_reviewers_stride_0_inout_is_refused_and_the_neighbouring_memory_is_not_written(monkeypatch):
    with pytest.raises(TilefishDeclarationError, match=r"parameter 'x' is annotated with View\[Count, f32\] \(strides \(0\), offset 0\).*Strided"):
        _fill(InOut[TensorType((Count,), f32, strides=(0,))], "fill_stride0")

    pool = torch.zeros(16384, device="cuda")
    small = pool[:1]
    expanded = small.expand(8192)  # one element of memory, 8192 elements of shape, stride 0
    assert expanded.stride() == (0,) and not expanded.is_contiguous()

    dense = _fill(InOut[Tensor[Count, f32]], "fill_dense")  # the declaration that is allowed: a launch checks contiguity
    with pytest.raises(TilefishPreconditionError, match="is not contiguous") as e:
        tilefish.run(dense, expanded)
    assert e.value.kind == "contiguous"

    monkeypatch.setattr(signature, "_require_contiguous", lambda *a: None)  # past the declaration, as the reviewer's was: the launch refuses it all the same
    bad = _fill(InOut[TensorType((Count,), f32, strides=(0,))], "fill_stride0_unchecked")
    with pytest.raises(TilefishPreconditionError, match="is not contiguous") as e:
        tilefish.run(bad, expanded)
    assert e.value.kind == "contiguous"
    torch.cuda.synchronize()
    assert int(torch.count_nonzero(pool)) == 0  # nothing was written: not the element, not its 8191 neighbours

    whole = pool[:8192]
    tilefish.run(dense, whole)  # a contiguous tensor runs
    torch.cuda.synchronize()
    assert bool((pool[:8192] == 7.0).all()) and int(torch.count_nonzero(pool[8192:])) == 0


@tilefish.extern(triton="gpu_kernels_p1:head_copy_kernel", num_warps=1, grid=lambda R: (R, 1))
def row_copy(x: Strided[Tensor[R, 1, 64, f32]], y: tilefish.Out[Tensor[R, 1, 64, f32]], x_s0: StrideOf("x", 0), x_s1: StrideOf("x", 1), D: const = 64): ...


def test_a_graph_parameter_with_strides_is_refused_and_a_graph_run_refuses_a_view():
    with pytest.raises(TilefishDeclarationError, match=r"graph 'reads_rows': parameter 'x' is annotated with View\[R, 1, 64, f32\] \(strides \(0, 64, 1\), offset 0\)"):

        @tilefish.graph
        def reads_rows(x: TensorType((R, 1, 64), f32, strides=(0, 64, 1))):  # the reviewer's graph: planned as row-major, read 64 rows of a tensor of one
            return row_copy(x)

    @tilefish.graph
    def copies_rows(x: Tensor[R, 1, 64, f32]):
        return row_copy(x)

    x = randn((64, 1, 64), seed=3)
    assert same_bytes(tilefish.run(copies_rows, x), x)
    one = randn((1, 1, 64), seed=4)
    with pytest.raises(TilefishPreconditionError, match="argument 'x' is not contiguous") as e:
        tilefish.run(copies_rows, one.expand(64, 1, 64))
    assert e.value.kind == "contiguous"
