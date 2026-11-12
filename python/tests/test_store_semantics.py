"""A store has copy semantics: it makes a fresh version of the target and never aliases its source."""

import copy
import pickle

import pytest

import tilefish
from tilefish import Dim, InOut, Tensor, TilefishTypeError, bf16

M = Dim("M")


@tilefish.library("inplace")
def inplace(x: InOut[Tensor[M, bf16]]): ...


@tilefish.library("copyto")
def copyto(src: Tensor[M, bf16], dst: InOut[Tensor[M, bf16]]): ...


def nodes(g, op):
    return [n for n in g.nodes if n.op == op]


def test_a_store_of_a_temp_then_an_in_place_update_of_the_temp_does_not_touch_the_target():
    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]], y: Tensor[M, bf16]):
        r = y + 1
        x[...] = r
        inplace(r)  # updates the temp after it was copied into x
        return x

    graph = g.graph
    add, store, bump = graph.nodes
    assert store.inputs == (graph.inputs[0], add.outputs[0])
    assert bump.inputs == (add.outputs[0],)  # the update is on the temp ...
    assert graph.outputs == (store.outputs[0],)  # ... and x is still the copy made by the store
    assert graph.outputs[0] is not bump.outputs[0]
    assert graph.finals == {"x": store.outputs[0]}
    assert "  final x = " + store.outputs[0].ref in str(graph)


def test_swapping_two_inout_params_through_a_temp_keeps_both_values():
    @tilefish.graph
    def swap(x: InOut[Tensor[M, bf16]], y: InOut[Tensor[M, bf16]]):
        tmp = x + 0
        x[...] = y
        y[...] = tmp
        return x, y

    graph = swap.graph
    x, y = graph.inputs
    tmp_node, store_x, store_y = graph.nodes
    assert store_x.inputs == (x, y)  # x receives the original y
    assert store_y.inputs == (y, tmp_node.outputs[0])  # y receives the original x, saved in the temp
    assert graph.outputs == (store_x.outputs[0], store_y.outputs[0])
    assert store_x.outputs[0] is not store_y.outputs[0]
    assert graph.finals == {"x": store_x.outputs[0], "y": store_y.outputs[0]}


def test_a_store_then_an_in_place_update_of_the_source_does_not_change_the_target():
    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]], y: InOut[Tensor[M, bf16]]):
        x[...] = y
        inplace(y)
        return x

    graph = g.graph
    store, bump = graph.nodes
    assert store.inputs == (graph.inputs[0], graph.inputs[1])
    assert bump.inputs == (graph.inputs[1],)
    assert graph.outputs == (store.outputs[0],)
    assert graph.finals == {"x": store.outputs[0], "y": bump.outputs[0]}


def test_a_store_then_an_in_place_update_of_the_target_does_not_change_the_source():
    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]], y: Tensor[M, bf16]):
        r = y * 2
        x[...] = r
        inplace(x)  # updates x's buffer, which holds a copy of r
        return r, x

    graph = g.graph
    mul, store, bump = graph.nodes
    assert bump.inputs == (store.outputs[0],)  # the update reads the stored copy, not r
    assert graph.outputs == (mul.outputs[0], bump.outputs[0])  # r is untouched; x is the updated copy
    assert graph.finals == {"x": bump.outputs[0]}


def test_a_read_only_input_never_becomes_writable_through_a_store():
    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]], w: Tensor[M, bf16]):
        x[...] = w  # copy w into x
        inplace(x)  # writes x's buffer, not w
        return x

    graph = g.graph
    x, w = graph.inputs
    store, bump = graph.nodes
    assert store.inputs == (x, w)
    assert bump.inputs == (store.outputs[0],) and bump.inputs[0] is not w
    assert w.live is w and w._latest is None  # w never got a version
    assert graph.outputs == (bump.outputs[0],)

    with pytest.raises(TilefishTypeError, match="inplace: argument 'x' is written in place but 'w' is a read-only input"):

        @tilefish.graph
        def direct(w: Tensor[M, bf16]):
            inplace(w)


def test_a_stored_copy_does_not_alias_its_source_in_later_calls():
    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]]):
        r = x + 0
        x[...] = r
        copyto(r, x)  # r is a snapshot and x holds a copy of it: two different tensors
        return x

    graph = g.graph
    assert graph.ops() == ["add", "store", "copyto"]
    copy_node = graph.nodes[2]
    assert copy_node.inputs[0] is graph.nodes[0].outputs[0]  # the snapshot
    assert copy_node.inputs[1] is graph.nodes[1].outputs[0]  # the stored copy


def test_reading_after_a_store_and_an_in_place_update_of_the_stored_temp():
    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]]):
        r = x + 1
        x[...] = r
        inplace(r)
        return x + 0

    graph = g.graph
    add1, store, bump, add2 = graph.nodes
    assert add2.inputs[0] is store.outputs[0]  # x reads its own version, not the updated r


def test_two_stores_in_a_row_keep_only_the_last_version():
    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]], y: Tensor[M, bf16], z: Tensor[M, bf16]):
        x[...] = y
        x[...] = z
        return x

    graph = g.graph
    first, second = graph.nodes
    assert first.inputs == (graph.inputs[0], graph.inputs[1])
    assert second.inputs == (graph.inputs[0], graph.inputs[2])
    assert graph.outputs == (second.outputs[0],) and graph.finals == {"x": second.outputs[0]}


def test_kernels_follow_the_same_rules():
    @tilefish.kernel
    def k(x: InOut[Tensor[M, bf16]], w: Tensor[M, bf16]):
        x[...] = w
        inplace(x)

    store, bump = k.graph.nodes
    assert bump.inputs == (store.outputs[0],)
    assert k.graph.finals == {"x": bump.outputs[0]}

    @tilefish.kernel
    def swap(x: InOut[Tensor[M, bf16]], y: InOut[Tensor[M, bf16]]):
        t = x + 0
        x[...] = y
        y[...] = t
        return x

    assert swap.graph.finals.keys() == {"x", "y"}
    assert swap.graph.outputs == (swap.graph.finals["x"],)


# -- the IR dump shows the final version of every modified InOut -----------------------------------------------------------


def test_the_dump_lists_final_versions_even_when_they_are_not_returned():
    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]], y: InOut[Tensor[M, bf16]], z: InOut[Tensor[M, bf16]]):
        x[...] = y + 1
        inplace(y)
        # z is never modified

    text = str(g.graph)
    graph = g.graph
    assert list(graph.finals) == ["x", "y"]
    assert f"  final x = {graph.finals['x'].ref}" in text
    assert f"  final y = {graph.finals['y'].ref}" in text
    assert "final z" not in text
    assert text.index("return") < text.index("final x") < text.index("final y")
    assert graph.outputs == ()  # nothing is returned, yet the modifications are visible


def test_unmodified_graphs_have_no_final_lines():
    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]]):
        return x

    assert g.graph.finals == {} and "final" not in str(g.graph)


# -- operators are pickled by reference, with a clear error when that is impossible --------------------------------------------


@tilefish.library("module_level_op")
def module_level_op(x: Tensor[M, bf16]) -> Tensor[M, bf16]: ...


def test_a_locally_defined_operator_cannot_be_pickled_and_says_why():
    @tilefish.library("local_op")
    def local_op(x: Tensor[M, bf16]) -> Tensor[M, bf16]: ...

    with pytest.raises(tilefish.TilefishPicklingError, match=r"cannot pickle library op 'local_op'.*module-level object.*defined inside a function"):
        pickle.dumps(local_op)
    with pytest.raises(pickle.PicklingError):  # still a PicklingError for code that catches that
        pickle.dumps(local_op)
    with pytest.raises(tilefish.TilefishError):
        pickle.dumps(local_op)
    assert copy.deepcopy(local_op) is local_op and copy.copy(local_op) is local_op  # copies stay fine


def test_a_module_level_operator_round_trips_by_reference():
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        assert pickle.loads(pickle.dumps(module_level_op, protocol=proto)) is module_level_op


def test_an_operator_that_was_rebound_cannot_be_pickled():
    import sys

    module = sys.modules[__name__]
    original = module.module_level_op
    try:
        module.module_level_op = object()  # the name no longer refers to the operator
        with pytest.raises(tilefish.TilefishPicklingError, match="reachable as"):
            pickle.dumps(original)
    finally:
        module.module_level_op = original
