"""A hand-written ``extern`` is taken to index with ``i32`` unless it says ``index_bits=64``: every tensor then has ``max_elements``.

A kernel that computes its offsets in 32 bits (``row * N_COLS + col`` on ``tl.int32`` values) reads and writes the wrong memory,
silently, on a tensor of ``2**31`` elements or more. A generated kernel has always carried a ``max_elements`` precondition for that;
a hand-written one said nothing about its indexing, so the in-tree ``rmsnorm`` (``row * N_COLS``) and ``mm`` passed every host check at
sizes where their offsets overflow. ``index_bits=32`` (the default) now declares ``max_elements`` for every pointer; a kernel that indexes with
``i64`` opts out with ``index_bits=64``. Python's ``replay`` checks it (here through ``runtime._prepare`` with fake tensors, the part of
a replay that decides a launch) and so does the Rust host (``crates/tilefish/tests/launch_checks.rs``, on the manifests of the
fixtures); the choice is part of the declaration digest.
"""

import pytest
from fake_torch import FakeTensor, prepare, verdict

import tilefish
from tilefish import Dim, Out, Tensor, bf16
from tilefish.aot import compile_specs, declaration_digest

LIMIT = 2**31 - 1
R = Dim("R")


@tilefish.extern(triton="idx_kernels:rows", grid=lambda R: (R,))
def narrow(x: Tensor[R, 1024, bf16], y: Out[Tensor[R, 1024, bf16]]): ...


@tilefish.extern(triton="idx_kernels:rows", grid=lambda R: (R,), index_bits=64)
def wide(x: Tensor[R, 1024, bf16], y: Out[Tensor[R, 1024, bf16]]): ...


def limits(op):
    (spec,) = compile_specs(op)
    return [(p.arg, p.shape, p.value) for p in spec.preconditions if p.kind == "max_elements"]


def call(op, rows):
    return prepare(op, [FakeTensor((rows, 1024), "bf16", 1 << 44)], next_slab=1 << 50)


def test_every_pointer_of_an_extern_has_the_limit_unless_it_indexes_with_i64():
    assert limits(narrow) == [("x", ("R", "1024"), LIMIT), ("y", ("R", "1024"), LIMIT)]
    assert limits(wide) == []
    assert narrow.index_bits == 32 and wide.index_bits == 64
    assert declaration_digest(narrow) != declaration_digest(wide)  # it is part of what the declaration says
    for op in (narrow, wide):
        (spec,) = compile_specs(op)
        spec.verify_against(op)  # and what a spec says is checked against the declaration


def test_python_refuses_a_tensor_that_i32_offsets_cannot_index_at_the_first_element_past_the_limit():
    prepared, error = call(narrow, 2**21 - 1)  # 2**21 - 1 rows of 1024: 2147482624 elements, under the limit
    assert error is None and verdict(prepared, error)["ok"]
    prepared, error = call(narrow, 2**21)  # 2**31 elements: one more than i32 can index
    assert prepared is None and error.kind == "max_elements" and error.arg == "x"
    assert "2147483648 elements" in str(error) and "i32 offsets" in str(error)


def test_a_kernel_that_declares_i64_indexing_is_not_limited():
    prepared, error = call(wide, 2**21)
    assert error is None and verdict(prepared, error)["grid"] == [2**21, 1, 1]
    prepared, error = call(wide, 2**23)
    assert error is None


def test_the_edge_is_the_limit_itself():
    # 2147483647 = 2**31 - 1 is not a multiple of 1024, so a row count cannot reach it exactly: the largest row count under it and the next
    assert call(narrow, LIMIT // 1024)[1] is None
    assert call(narrow, LIMIT // 1024 + 1)[1].kind == "max_elements"


@pytest.mark.parametrize("bad", [16, 0, 128, True, False, "32", None, 32.0, 64.0])
def test_index_bits_is_the_integer_32_or_64(bad):
    with pytest.raises(tilefish.TilefishDeclarationError, match="index_bits must be 32"):
        tilefish.extern(triton="idx_kernels:rows", grid=lambda R: (R,), index_bits=bad)


def test_the_in_tree_externs_that_index_with_i32_pass_no_check_at_sizes_where_they_overflow_no_more():
    """The reviewer's repro: ``rmsnorm`` (``row * N_COLS``) and ``mm`` were launched at sizes where i32 offsets overflow."""
    from host_ops import ops

    declared = ops()
    for key in ("rmsnorm", "mm", "gelu_erf", "axpy"):
        for spec in compile_specs(declared[key]):
            pointers = {a.name for a in spec.args if a.kind == "pointer"}
            assert {p.arg for p in spec.preconditions if p.kind == "max_elements"} == pointers, key
    rows = (1 << 20) - 1  # rmsnorm: x is (Rows, 2048)
    ok = prepare(declared["rmsnorm"], [FakeTensor((rows, 2048), "bf16", 1 << 44), FakeTensor((2048,), "bf16", 1 << 50)], next_slab=1 << 52)
    assert ok[1] is None
    too_many = prepare(
        declared["rmsnorm"], [FakeTensor((rows + 1, 2048), "bf16", 1 << 44), FakeTensor((2048,), "bf16", 1 << 50)], next_slab=1 << 52
    )
    assert too_many[0] is None and too_many[1].kind == "max_elements" and too_many[1].arg == "x_ptr"
