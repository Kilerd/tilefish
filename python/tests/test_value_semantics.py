"""Values have value semantics: an in-place call consumes the version it reads, and a no-op cast is a read.

A parameter's name follows its newest version. Every other value (a temporary, a split chunk, an older
version) is immutable: once an in-place call consumed it, reading it raises instead of silently
reading storage that was updated.
"""

import pytest

import tilefish
from tilefish import Dim, InOut, Tensor, TilefishTypeError, bf16, f32

M = Dim("M")


@tilefish.library("inplace")
def inplace(x: InOut[Tensor[M, bf16]]): ...


@tilefish.library("inplace2")
def inplace2(x: InOut[Tensor[2, 8, bf16]]): ...


@tilefish.library("accumulate")
def accumulate(acc: InOut[Tensor[M, bf16]], x: Tensor[M, bf16]): ...


@tilefish.library("combine")
def combine(a: Tensor[M, bf16], b: Tensor[M, bf16]) -> Tensor[M, bf16]: ...


STALE = r"reads %\w+, which the in-place call to '(\w+)' \(InOut parameter '(\w+)'\) already consumed"


def new_version(node):
    """The InOut version an in-place call produced (the last output of the node)."""
    return node.outputs[-1]


# -- reading a value that an in-place call consumed ---------------------------------------------------------------------------------


def test_a_temporary_passed_to_an_inout_param_cannot_be_read_afterwards():
    with pytest.raises(TilefishTypeError, match=STALE) as info:

        @tilefish.graph
        def g(x: Tensor[M, bf16]):
            r = x + 1
            inplace(r)
            return r * 2  # r was consumed: this is not the updated tensor

    text = str(info.value)
    assert "in graph 'g': mul (*): left operand reads %0" in text
    assert "in-place call to 'inplace' (InOut parameter 'x')" in text  # names the op that consumed it
    assert "`.live`" in text and "(here %1)" in text  # and says how to get the new value


@pytest.mark.parametrize(
    "use",
    [
        lambda r: r + 1,
        lambda r: 1 + r,
        lambda r: r * r,
        lambda r: tilefish.silu(r),
        lambda r: tilefish.sum(r),
        lambda r: r.to(f32),
        lambda r: r.to(bf16),  # a no-op cast is a read too
        lambda r: r.split(2, axis=0),
        lambda r: combine(r, r),
        lambda r: inplace(r),  # a second in-place call on the consumed name
        lambda r: accumulate(r, r),
    ],
    ids="add radd mul silu sum cast noop-cast split call inplace-again inout-and-input".split(),
)
def test_every_kind_of_read_of_a_consumed_value_raises(use):
    with pytest.raises(TilefishTypeError, match=STALE):

        @tilefish.graph
        def g(x: Tensor[M, bf16]):
            r = x + 1
            inplace(r)
            use(r)


def test_returning_or_storing_a_consumed_value_raises_too():
    with pytest.raises(TilefishTypeError, match=r"graph 'ret': the returned value reads %0, which the in-place call to 'inplace'"):

        @tilefish.graph
        def ret(x: Tensor[M, bf16]):
            r = x + 1
            inplace(r)
            return r

    with pytest.raises(TilefishTypeError, match=r"store: the stored value reads %0, which the in-place call to 'inplace'"):

        @tilefish.graph
        def put(x: InOut[Tensor[M, bf16]], y: Tensor[M, bf16]):
            r = y + 1
            inplace(r)
            x[...] = r


def test_a_kernel_checks_the_same():
    with pytest.raises(TilefishTypeError, match=r"in kernel 'k': .*already consumed"):

        @tilefish.kernel
        def k(x: Tensor[M, bf16], y: tilefish.Out[Tensor[M, bf16]]):
            r = x + 1
            inplace(r)
            y[...] = r


def test_the_updated_tensor_is_its_live_version():
    @tilefish.graph
    def g(x: Tensor[M, bf16]):
        r = x + 1
        inplace(r)
        newest = r.live  # the explicit way to the new version
        s = newest * 2
        inplace(newest)  # and it can be updated again
        return s, newest.live

    add, bump1, mul, bump2 = g.graph.nodes
    assert bump1.inputs == (add.outputs[0],)
    assert mul.inputs[0] is new_version(bump1)
    assert bump2.inputs == (new_version(bump1),)
    assert g.graph.outputs == (mul.outputs[0], new_version(bump2))


def test_stale_older_versions_are_rejected():
    with pytest.raises(TilefishTypeError, match=r"reads %0, which the in-place call to 'inplace' \(InOut parameter 'x'\) already consumed.*\(here %1\)"):

        @tilefish.graph
        def g(x: InOut[Tensor[M, bf16]]):
            inplace(x)
            v1 = x.live
            inplace(x)
            return v1 + 1


def test_parameters_keep_reading_their_newest_version():
    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]], y: Tensor[M, bf16]):
        inplace(x)
        a = x + 1  # reads the version the call produced
        inplace(x)
        b = x + 1  # reads the next one
        return a, b

    bump1, a, bump2, b = g.graph.nodes
    assert a.inputs[0] is new_version(bump1) and b.inputs[0] is new_version(bump2)
    assert bump2.inputs == (new_version(bump1),)


def test_a_value_the_in_place_call_did_not_consume_stays_readable():
    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]], y: Tensor[M, bf16]):
        t = y + 1
        accumulate(x, t)  # t is only read
        inplace(x)
        return t + 1, y * 2  # t and y are untouched

    assert g.graph.ops() == ["add", "accumulate", "inplace", "add", "mul"]


def test_a_failed_call_consumes_nothing():
    @tilefish.graph
    def g(x: Tensor[M, bf16]):
        r = x + 1
        with pytest.raises(TilefishTypeError, match="expected dtype"):
            inplace(r.to(f32))  # rejected before anything is recorded
        return r + 1

    assert g.graph.ops() == ["add", "cast", "add"]


# -- temporaries are versioned like parameters ---------------------------------------------------------------------------------------------


def test_a_temporary_gets_a_version_per_in_place_call_like_a_parameter():
    @tilefish.graph
    def g(p: InOut[Tensor[M, bf16]], y: Tensor[M, bf16]):
        t = y + 1
        accumulate(p, t)
        inplace(p)
        inplace(t)
        inplace(t.live)
        return p, t.live

    add, acc, bump_p, bump_t1, bump_t2 = g.graph.nodes
    # the parameter: one version per call, chained through the inputs
    assert acc.inputs == (g.graph.inputs[0], add.outputs[0])
    assert bump_p.inputs == (new_version(acc),)
    # the temporary: the same
    assert bump_t1.inputs == (add.outputs[0],)
    assert bump_t2.inputs == (new_version(bump_t1),)
    assert g.graph.outputs == (new_version(bump_p), new_version(bump_t2))
    assert bump_t1.attrs["inout"] == bump_t2.attrs["inout"] == ("x",)
    assert g.graph.finals == {"p": new_version(bump_p)}  # only parameters are reported


# -- an in-place update of a split chunk does not change the parent --------------------------------------------------------------------------


def test_updating_a_chunk_in_place_leaves_the_parent_unchanged():
    @tilefish.graph
    def g(p: InOut[Tensor[4, 8, bf16]]):
        a, b = p.split(2, axis=0)
        inplace2(a)
        return p + 0, b + 0  # the parent and the sibling are as they were

    split, bump, add_p, add_b = g.graph.nodes
    assert bump.inputs == (split.outputs[0],)
    assert add_p.inputs[0] is g.graph.inputs[0]  # the parent's own version
    assert add_b.inputs[0] is split.outputs[1]
    assert g.graph.finals == {}  # p was never modified, only a chunk of it was
    assert "final" not in str(g.graph)


def test_the_updated_chunk_is_a_new_version_of_the_chunk_only():
    @tilefish.graph
    def g(x: Tensor[4, 8, bf16]):
        a, b = x.split(2, axis=0)
        inplace2(a)
        return a.live, b

    split, bump = g.graph.nodes
    assert g.graph.outputs == (new_version(bump), split.outputs[1])
    with pytest.raises(TilefishTypeError, match=STALE):

        @tilefish.graph
        def h(x: Tensor[4, 8, bf16]):
            a, b = x.split(2, axis=0)
            inplace2(a)
            return a + 0  # the chunk itself was consumed


def test_a_kernel_that_only_updates_a_chunk_of_an_inout_param_has_not_written_it():
    with pytest.raises(TilefishTypeError, match="InOut parameter 'p' is never written"):

        @tilefish.kernel
        def k(p: InOut[Tensor[4, 8, bf16]]):
            a, _ = p.split(2, axis=0)
            inplace2(a)  # a is a value: p is untouched


def test_the_docs_say_so():
    for doc in (tilefish.InOut.__doc__, tilefish.ops.split.__doc__, tilefish.Node.__doc__):
        doc = " ".join(doc.split())
        assert "does not change" in doc or "does **not** change" in doc or "not modified" in doc, doc
    assert "consume" in tilefish.InOut.__doc__ and "split chunk" in tilefish.InOut.__doc__
    assert "value semantics" in tilefish.Value.__doc__


# -- a no-op cast reads the value at that point ------------------------------------------------------------------------------------------------


def test_a_noop_cast_is_a_read_at_that_point_before_the_first_store():
    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]], y: Tensor[M, bf16], z: Tensor[M, bf16]):
        c = x.to(bf16)  # no-op: the value x has right now
        x[...] = y  # a store gives x a new version ...
        a = c + 0  # ... which c does not follow
        x[...] = z
        b = c + 0  # nor this one
        return a, b, x

    graph = g.graph
    x = graph.inputs[0]
    store1, add_a, store2, add_b = graph.nodes
    assert add_a.inputs[0] is x and add_b.inputs[0] is x  # both read the parameter as it was passed in
    assert store1.inputs[0] is x and store2.inputs[0] is store1.outputs[0]
    assert graph.outputs[2] is store2.outputs[0]  # the name x reads the newest version


def test_a_noop_cast_is_a_read_at_that_point_after_the_first_store():
    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]], y: Tensor[M, bf16], z: Tensor[M, bf16]):
        x[...] = y
        c = x.to(bf16)  # no-op: the version the store made
        x[...] = z  # a later store ...
        a = c + 0  # ... leaves it as it was
        return a, x

    store1, store2, add_a = g.graph.nodes[0], g.graph.nodes[1], g.graph.nodes[2]
    assert add_a.inputs[0] is store1.outputs[0]
    assert g.graph.outputs[1] is store2.outputs[0]


@pytest.mark.parametrize("first_store", [False, True], ids=["before-first-store", "after-first-store"])
def test_a_noop_cast_does_not_follow_a_later_in_place_update(first_store):
    with pytest.raises(TilefishTypeError, match=STALE) as info:

        @tilefish.graph
        def g(x: InOut[Tensor[M, bf16]], y: Tensor[M, bf16]):
            if first_store:
                x[...] = y
            c = x.to(bf16)
            inplace(x)  # consumes the version c read
            return c + 0  # c does not silently become the update

    assert "in-place call to 'inplace' (InOut parameter 'x')" in str(info.value)
    assert "`.live`" in str(info.value)


@pytest.mark.parametrize("first_store", [False, True], ids=["before-first-store", "after-first-store"])
def test_the_updated_tensor_after_a_noop_cast_is_reached_through_live_or_the_name(first_store):
    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]], y: Tensor[M, bf16]):
        if first_store:
            x[...] = y
        c = x.to(bf16)
        inplace(x)
        return c.live + 0, x + 0

    *_, bump, add_c, add_x = g.graph.nodes
    assert add_c.inputs[0] is new_version(bump) and add_x.inputs[0] is new_version(bump)


def test_a_noop_cast_is_the_same_before_and_after_the_first_store():
    """The cast of a parameter is a reference to its current version in both cases, never the parameter itself."""

    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]], y: Tensor[M, bf16]):
        before = x.to(bf16)
        assert before is not x  # not the parameter, whose name follows later updates
        assert before.type == x.type and before.ref == x.ref and not before.is_param
        x[...] = y
        after = x.to(bf16)
        assert after is x.live  # already a version, immutable like any value
        assert after.live is after
        return before + 0, after + 0

    store, add_before, add_after = g.graph.nodes
    assert add_before.inputs[0] is g.graph.inputs[0]  # the parameter as it was passed in
    assert add_after.inputs[0] is store.outputs[0]  # the version the store made


def test_a_noop_cast_of_the_cast_and_other_uses():
    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]], y: Tensor[M, bf16]):
        c = x.to(bf16)
        d = c.to(bf16)  # a cast of a reference is a reference to the same version
        e = combine(c, d)  # both read the same version of x
        return e, d.to(f32)

    (combine_node, _cast) = g.graph.nodes
    assert combine_node.inputs == (g.graph.inputs[0], g.graph.inputs[0])


def test_returning_a_noop_cast_returns_the_version_it_read():
    @tilefish.graph
    def g(x: InOut[Tensor[M, bf16]]):
        return x.to(bf16)

    assert g.graph.outputs == (g.graph.inputs[0],)
    assert g.graph.finals == {}


def test_updating_through_a_noop_cast_updates_the_parameter():
    @tilefish.kernel
    def k(x: InOut[Tensor[M, bf16]]):
        c = x.to(bf16)
        inplace(c)  # c is the version x has now, so this is an in-place update of x

    (bump,) = k.graph.nodes
    assert bump.inputs == (k.graph.inputs[0],)
    assert k.graph.finals == {"x": new_version(bump)}  # and x counts as written


def test_storing_a_noop_cast_of_the_parameter_is_a_self_store():
    with pytest.raises(TilefishTypeError, match="InOut parameter 'x' is never written"):

        @tilefish.kernel
        def k(x: InOut[Tensor[M, bf16]]):
            x[...] = x.to(bf16)


def test_a_noop_cast_of_a_read_only_input_or_a_temporary_is_the_value_itself():
    @tilefish.graph
    def g(w: Tensor[M, bf16], x: InOut[Tensor[M, bf16]]):
        t = w + 1
        assert w.to(bf16) is w and t.to(bf16) is t  # neither is ever replaced
        return t.to(bf16)

    assert g.graph.outputs == (g.graph.nodes[0].outputs[0],)

    with pytest.raises(TilefishTypeError, match="is written in place but 'w' is a read-only input"):

        @tilefish.graph
        def bad(w: Tensor[M, bf16]):
            inplace(w.to(bf16))
