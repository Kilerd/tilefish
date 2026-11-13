"""A dimension that only an ``Out`` tensor determines: an extern that writes a ``[B]`` mask and reads a device scalar.

No input of ``key_mask(mask: Out[Tensor[B, bf16]], n_valid: DevScalar[i32], b: B)`` has ``B``. The rule:

* **a declaration** of an extern may have such a dimension (``Signature.output_dims``); a kernel, a library op and the offset dimension of a
  ``Strided[...]`` parameter may not;
* **a launch** binds it from the ``Out`` tensor it is given, after the dimensions the inputs bind, and checks it like any other (its buckets, its
  ``divisible_by``, the runtime scalar that carries it). A plan step and a Rust launch are always given their ``Out`` tensors; ``tilefish.run`` is given
  them with ``outputs=``, and without them says so;
* **a call in a graph** binds it to the graph's dimension that is the same ``Dim``: a plan is made per bucket of the graph's dimensions, so every step
  knows it, and binds it from the step's ``Out`` buffer. A graph that has no such dimension is refused at the call.

Nothing here needs a GPU (``test_gpu_output_dims.py`` runs the kernels).
"""

import pytest
from fake_torch import FakeTensor, prepare, verdict

import tilefish
from tilefish import (
    DevScalar,
    Dim,
    Out,
    StrideOf,
    Strided,
    Tensor,
    TilefishDeclarationError,
    TilefishPreconditionError,
    TilefishRuntimeError,
    TilefishTypeError,
    bf16,
    const,
    f32,
    i32,
)
from tilefish.aot import compile_specs

B = Dim("B", buckets=[8, 24])
N = Dim("N")


@tilefish.extern(triton="gpu_kernels_p4:key_mask_kernel", num_warps=1, grid=lambda B: (tilefish.cdiv(B, 128),))
def key_mask(mask: Out[Tensor[B, bf16]], n_valid: DevScalar[i32], b: B, BLOCK: const = 128): ...


@tilefish.extern(triton="gpu_kernels_p4:key_mask_kernel", num_warps=1, grid=lambda N: (tilefish.cdiv(N, 128),), name="key_mask_any")
def key_mask_any(mask: Out[Tensor[N, bf16]], n_valid: DevScalar[i32], b: N, BLOCK: const = 128): ...


@tilefish.extern(triton="gpu_kernels_p4:apply_mask_kernel", num_warps=1, grid=lambda B: (B,))
def apply_mask(x: Tensor[B, 64, f32], m: Tensor[B, bf16], y: Out[Tensor[B, 64, f32]], D: const = 64): ...


# -- the declaration ----------------------------------------------------------------------------------------------------------------------------


def test_an_extern_may_have_a_dimension_only_its_out_tensors_determine():
    assert key_mask.signature.output_dims == frozenset({B})
    assert key_mask_any.signature.output_dims == frozenset({N})
    assert apply_mask.signature.output_dims == frozenset()
    (spec,) = compile_specs(key_mask)  # one binary: B is a runtime scalar, its buckets a constraint the host checks
    assert spec.dims == {"B": {"buckets": [8, 24]}} and spec.args[2].dim == "B" and spec.grid == "(cdiv(B, 128),)"


def test_an_out_dimension_may_be_an_expression_and_carry_a_scalar():
    @tilefish.extern(triton="m:k", grid=lambda N: (N,), name="pairs")
    def pairs(y: Out[Tensor[2 * N, f32]], n: DevScalar[i32], half: N): ...

    assert pairs.signature.output_dims == frozenset({N})


def test_a_kernel_and_the_offset_of_a_view_may_not_have_one():
    with pytest.raises(TilefishDeclarationError, match=r"Out parameter 'y' mentions dimension\(s\) N that no input tensor determines"):

        @tilefish.kernel
        def k(n: DevScalar[i32], y: Out[Tensor[N, f32]]):
            y[...] = tilefish.load(n).to(f32)

    xo = Dim("x_offset")
    with pytest.raises(TilefishDeclarationError, match=r"x_offset that no input tensor determines \(the offset dimension of a Strided"):

        @tilefish.extern(triton="m:k", grid=(1,), name="by_offset")
        def by_offset(x: Strided[Tensor[N, 4, f32]], y: Out[Tensor[xo, f32]], x_s0: StrideOf("x", 0)): ...

    with pytest.raises(TilefishDeclarationError, match=r"runtime parameter 'c' mentions dimension\(s\) C that no input tensor determines"):

        @tilefish.extern(triton="m:k", grid=(1,), name="nowhere")
        def nowhere(n: DevScalar[i32], y: Out[Tensor[N, f32]], c: Dim("C")): ...


# -- a launch: from the Out tensor it is given ---------------------------------------------------------------------------------------------------

SLAB = 1 << 36
N_PTR, OUT_PTR = 1 << 44, (1 << 44) + SLAB


def launch(op, mask_shape, n_shape=(), mask_dtype="bf16", outputs=True):
    n = FakeTensor(n_shape, "i32", N_PTR)
    out = FakeTensor(mask_shape, mask_dtype, OUT_PTR)
    return prepare(op, [n], next_slab=OUT_PTR + SLAB, outputs=[out] if outputs else None)


@pytest.mark.parametrize("b", [8, 24])
def test_a_launch_binds_the_dimension_from_the_out_tensor(b):
    prepared, error = launch(key_mask, (b,))
    assert error is None and prepared.scalars == {"b": b} and prepared.grid == (1,)
    assert verdict(prepared, None)["params"] == [OUT_PTR, N_PTR, b, 0, 0]
    assert prepared.results[0].data_ptr() == OUT_PTR  # the tensor given is the result


def test_the_dimension_bound_from_the_out_tensor_is_held_to_its_buckets():
    _, error = launch(key_mask, (16,))
    assert isinstance(error, TilefishPreconditionError) and error.kind == "bucket" and "dimension B is 16" in str(error)
    prepared, error = launch(key_mask_any, (1000,))  # unbucketed: any size
    assert error is None and prepared.scalars == {"b": 1000} and prepared.grid == (8,)


@pytest.mark.parametrize(
    "shape, dtype, kind",
    [((24, 1), "bf16", "rank"), ((), "bf16", "rank"), ((24,), "f32", "dtype"), ((0,), "bf16", "shape")],
)
def test_an_out_tensor_that_is_not_what_its_parameter_says_is_refused(shape, dtype, kind):
    _, error = launch(key_mask_any, shape, mask_dtype=dtype)
    assert isinstance(error, TilefishPreconditionError) and error.kind == kind, error


def test_an_expression_is_solved_from_the_out_tensor():
    @tilefish.extern(triton="m:k", grid=lambda N: (N,), name="pairs")
    def pairs(y: Out[Tensor[2 * N, f32]], n: DevScalar[i32], half: N): ...

    out = lambda size: FakeTensor((size,), "f32", OUT_PTR)  # noqa: E731
    prepared, error = prepare(pairs, [FakeTensor((), "i32", N_PTR)], next_slab=OUT_PTR + SLAB, outputs=[out(6)])
    assert error is None and prepared.scalars == {"half": 3}
    _, error = prepare(pairs, [FakeTensor((), "i32", N_PTR)], next_slab=OUT_PTR + SLAB, outputs=out(7))
    assert isinstance(error, TilefishPreconditionError) and error.kind == "shape" and "would need N = 7/2" in str(error)


def test_without_its_out_tensor_a_call_says_how_to_give_it():
    _, error = launch(key_mask, (8,), outputs=False)
    assert isinstance(error, TilefishRuntimeError) and "dimension B is not determined by the arguments" in str(error) and "outputs=" in str(error)


def test_outputs_must_be_one_per_result():
    n = FakeTensor((), "i32", N_PTR)
    out = FakeTensor((8,), "bf16", OUT_PTR)
    _, error = prepare(key_mask, [n], next_slab=OUT_PTR + SLAB, outputs=[out, out])
    assert isinstance(error, TilefishRuntimeError) and "outputs= gives 2 tensor(s), but 'key_mask' has 1 result(s) (mask)" in str(error)


def test_outputs_of_a_call_whose_inputs_bind_everything_are_held_to_the_shape():
    """``outputs=`` of any kernel: written instead of allocated, and refused when it is not the tensor the call would allocate."""
    x = FakeTensor((8, 64), "f32", N_PTR)
    m = FakeTensor((8,), "bf16", N_PTR + SLAB)
    prepared, error = prepare(apply_mask, [x, m], next_slab=N_PTR + 4 * SLAB, outputs=FakeTensor((8, 64), "f32", N_PTR + 2 * SLAB))
    assert error is None and prepared.tensors["y"].data_ptr() == N_PTR + 2 * SLAB
    _, error = prepare(apply_mask, [x, m], next_slab=N_PTR + 4 * SLAB, outputs=FakeTensor((8, 32), "f32", N_PTR + 2 * SLAB))
    assert isinstance(error, TilefishPreconditionError) and error.kind == "static_size" and "argument 'y'" in str(error)


# -- a call in a graph: the graph's dimension that is the same Dim --------------------------------------------------------------------------------


def test_a_graph_call_binds_it_to_the_graphs_dimension_and_a_plan_binds_it_from_the_step():
    @tilefish.graph
    def mask_rows(x: Tensor[B, 64, f32], n: DevScalar[i32]) -> Tensor[B, 64, f32]:
        m = key_mask(n)
        return apply_mask(x, m)

    assert str(mask_rows.graph.nodes[0].outputs[0].type) == "Tensor[B, bf16]"
    plan = tilefish.plan(mask_rows)
    assert plan.dims == {"B": (8, 24)}
    for b in (8, 24):
        steps = plan.steps(B=b)
        assert [s["entry"] for s in steps] == ["key_mask", "apply_mask"]
        assert steps[0]["args"] == [{"name": "mask", "binding": "t0"}, {"name": "n_valid", "binding": "n"}]
    mask = next(buf for buf in plan.buffers if buf["name"] == "t0")
    assert mask["role"] == "intermediate" and mask["shape"] == ["B"] and mask["bytes"] == 48


def test_a_graph_without_the_dimension_cannot_make_the_call():
    with pytest.raises(TilefishTypeError, match=r"key_mask: the dimension Dim\('B', buckets=\[8, 24\]\) of Out parameter 'mask' is determined by no argument") as e:

        @tilefish.graph
        def other(x: Tensor[N, 64, f32], n: DevScalar[i32]):
            return key_mask(n)

    assert "graph 'other' has none" in str(e.value)
    namesake = Dim("B", buckets=[8, 24, 32])
    with pytest.raises(TilefishTypeError, match=r"has Dim\('B', buckets=\[8, 24, 32\]\), which is not the same dimension"):

        @tilefish.graph
        def close(x: Tensor[namesake, 64, f32], n: DevScalar[i32]):
            return key_mask(n)


def test_an_unbucketed_out_dimension_of_a_graph_dimension_is_planned_per_bucket():
    @tilefish.graph
    def any_mask(x: Tensor[N, 64, f32], n: DevScalar[i32]):
        return key_mask_any(n)

    plan = tilefish.plan(any_mask, buckets={"N": [4, 100]})
    assert [plan.steps(N=v)[0]["entry"] for v in (4, 100)] == ["key_mask_any", "key_mask_any"]
    assert next(buf for buf in plan.buffers if buf["name"] == "out0")["shape"] == ["N"]
