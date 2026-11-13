"""Launch plans (``tilefish.plan``): the steps, the buffers and the arena of a graph, with no GPU.

A plan is made by walking the traced graph once: every call to a declared operator is a step (one launch of a manifest entry), the view operations are
views of a buffer (no step), and every other buffer is an intermediate in one arena at a 256-byte aligned offset that is the same in every bucket. These tests
make the plan of a generic transformer block (``plan_model``: lowered row and elementwise kernels, pinned cuBLASLt GEMMs, a hand-written attention that reads
strided views and a device scalar) and of small graphs that have an in-place tensor, and check what the plan says and what it refuses.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import json

import plan_model as pm
import pytest

import tilefish
from tilefish import Dim, InOut, Out, Tensor, TilefishPlanError, bf16, f32, i32
from tilefish import _plan


@pytest.fixture(scope="module")
def made():
    return tilefish.plan(pm.block)


def arena_buffers(p):
    return [b for b in p.buffers if b["offset"] is not None]


def test_a_plan_has_one_list_of_steps_per_bucket_and_the_steps_are_the_calls_in_the_order_of_the_ir(made):
    assert made.buckets() == [{"M": 32}, {"M": 64}, {"M": 128}]
    for m in pm.BUCKETS:
        steps = made.steps(M=m)
        assert [s["entry"] for s in steps] == [
            "rmsnorm", f"gemm_qkv.M{m}", "attention", f"gemm_o.M{m}", "add_rmsnorm", f"gemm_up.M{m}", "gelu", f"gemm_down.M{m}", "add",
        ]
        assert [s["kind"] for s in steps] == ["kernel", "library", "kernel", "library", "kernel", "library", "kernel", "library", "kernel"]
    # the views are not steps: nine calls, and the IR has the reshape, three selects and a second reshape between them
    assert len(pm.block.graph.nodes) == 9 + 5


def test_the_buffers_are_the_parameters_then_the_results_and_one_workspace(made):
    names = [(b["name"], b["role"]) for b in made.buffers]
    assert names[:9] == [
        ("x", "input"), ("n", "devscalar"), ("w_ln1", "weight"), ("w_qkv", "weight"), ("b_qkv", "weight"), ("w_o", "weight"),
        ("w_ln2", "weight"), ("w_up", "weight"), ("w_down", "weight"),
    ]
    assert [n for n, role in names[9:] if role == "intermediate"] == [f"t{i}" for i in range(9)]
    assert ("out0", "output") in names and names[-1] == ("workspace", "workspace")
    assert made.outputs == ["out0"]


def test_a_weight_is_a_parameter_whose_shape_mentions_no_dimension(made):
    roles = {b["name"]: b["role"] for b in made.buffers}
    assert roles["x"] == "input" and roles["w_qkv"] == "weight" and roles["b_qkv"] == "weight"


def test_q_k_and_v_are_views_of_the_fused_output_and_the_attention_output_is_reshaped_for_the_next_gemm(made):
    for m in pm.BUCKETS:
        step = {s["entry"]: s for s in made.steps(M=m)}["attention"]
        args = {a["name"]: a for a in step["args"]}
        fused = args["q"]["binding"]
        assert {args[n]["binding"] for n in "qkv"} == {fused}
        for i, name in enumerate("qkv"):
            assert args[name]["view"] == {"shape": [m, 4, 64], "strides": [768, 64, 1], "offset": 256 * i}
        assert "view" not in args["o"] and "view" not in args["n_valid"]  # a result and a device scalar are bound whole
        assert args["n_valid"]["binding"] == "n"
        gemm = {s["entry"]: s for s in made.steps(M=m)}[f"gemm_o.M{m}"]
        a = {x["name"]: x for x in gemm["args"]}["a"]
        assert a["binding"] == args["o"]["binding"] and a["view"] == {"shape": [m, 256], "strides": [256, 1], "offset": 0}


def test_every_library_step_has_the_one_workspace_and_no_kernel_step_has_one(made):
    for step in made.steps(M=64):
        assert ("workspace" in step) == (step["kind"] == "library")
        assert step.get("workspace", "workspace") == "workspace"
    ws = [b for b in made.buffers if b["role"] == "workspace"]
    assert len(ws) == 1 and ws[0]["bytes"] == 8_519_680 and ws[0]["dtype"] == "u8" and ws[0]["shape"] == [8_519_680]


def test_the_arena_offsets_are_aligned_and_do_not_overlap_without_reuse(made):
    regions = sorted((b["offset"], b["offset"] + _plan.align_up(b["bytes"]), b["name"]) for b in arena_buffers(made))
    assert all(lo % 256 == 0 for lo, _, _ in regions)
    assert all(hi <= nxt for (_, hi, _), (nxt, _, _) in zip(regions, regions[1:]))
    assert made.arena_bytes == regions[-1][1] == made.arena_bytes_without_reuse
    assert made.reuse is False


def test_a_buffer_is_sized_for_the_largest_bucket_and_its_offset_does_not_depend_on_the_bucket(made):
    by_name = {b["name"]: b for b in made.buffers}
    assert by_name["t1"]["bytes"] == 128 * 768 * 2 and by_name["t1"]["shape"] == ["M", 768]
    assert by_name["x"]["bytes"] == 128 * 256 * 2
    assert by_name["n"]["bytes"] == 4 and by_name["n"]["shape"] == [] and by_name["n"]["offset"] % 256 == 0
    # the offset is a field of the buffer, not of a bucket: every bucket's steps name the same buffers
    skeletons = {tuple((a["name"], a["binding"]) for s in made.steps(M=m) for a in s["args"]) for m in pm.BUCKETS}
    assert len(skeletons) == 1


def test_reuse_shares_bytes_only_between_buffers_no_step_uses_together():
    plain, reused = tilefish.plan(pm.block), tilefish.plan(pm.block, reuse=True)
    assert reused.reuse is True
    assert reused.arena_bytes < plain.arena_bytes == reused.arena_bytes_without_reuse
    assert reused.arena_bytes == plain.arena_bytes_with_reuse and reused.arena_bytes_without_reuse == plain.arena_bytes
    # the same steps either way: reuse changes where a buffer is, nothing else
    for m in pm.BUCKETS:
        assert reused.steps(M=m) == plain.steps(M=m)
    live = {}
    for s_index, s in enumerate(reused.steps(M=64)):
        for a in s["args"]:
            lo, hi = live.get(a["binding"], (s_index, s_index))
            live[a["binding"]] = (min(lo, s_index), max(hi, s_index))
    regions = {b["name"]: (b["offset"], b["offset"] + _plan.align_up(b["bytes"])) for b in arena_buffers(reused)}
    shared = 0
    for a in regions:
        for b in regions:
            if a < b and regions[a][0] < regions[b][1] and regions[b][0] < regions[a][1]:
                shared += 1
                (alo, ahi), (blo, bhi) = live[a], live[b]
                assert ahi < blo or bhi < alo, (a, b)  # no step uses both
    assert shared > 0
    # a device scalar and the workspace live for the whole plan: nothing shares their bytes
    for special in ("n", "workspace"):
        for other in regions:
            if other != special:
                assert regions[special][1] <= regions[other][0] or regions[other][1] <= regions[special][0]


def test_the_plan_document_is_a_function_of_the_graph_and_its_digest_follows_every_field(made):
    again = tilefish.plan(pm.block)
    assert again.doc == made.doc and again.digest == made.digest and again.to_json() == made.to_json()
    assert made.doc["plan_digest"] == made.digest == _plan.plan_digest(made.doc)
    assert tilefish.plan(pm.block, reuse=True).digest != made.digest
    assert tilefish.plan(pm.block, {"M": [32, 64]}).digest != made.digest

    def changed(path):
        doc = made.doc
        node = doc
        for key in path[:-1]:
            node = node[key]
        old = node[path[-1]]
        node[path[-1]] = old + 1 if isinstance(old, int) and not isinstance(old, bool) else (str(old) + "x" if old is None or isinstance(old, str) else [old, 0])
        return _plan.plan_digest(doc)

    for path in (
        ["graph"], ["arena_bytes"], ["dims", "M"], ["buffers", 3, "offset"], ["buffers", 0, "name"], ["buckets", 1, "steps", 0, "entry"],
        ["buckets", 2, "steps", 2, "args", 0, "view", "offset"], ["buckets", 0, "values"], ["manifest_digest"], ["arch"], ["outputs"],
    ):
        assert changed(path) != made.digest, path
    # the version that wrote the file is not digested, like the manifest's
    doc = made.doc
    doc["tilefish"] = "9.9.9"
    assert _plan.plan_digest(doc) == made.digest


def test_buckets_can_be_a_subset_of_the_declared_ones_and_not_anything_else():
    p = tilefish.plan(pm.block, buckets={"M": [64]})
    assert p.dims == {"M": (64,)} and p.buckets() == [{"M": 64}]
    assert tilefish.plan(pm.block, buckets={pm.M: [128, 32]}).dims == {"M": (32, 128)}
    with pytest.raises(TilefishPlanError, match="not declared with"):
        tilefish.plan(pm.block, buckets={"M": [48]})
    with pytest.raises(TilefishPlanError, match="not dimensions of the graph"):
        tilefish.plan(pm.block, buckets={"Q": [1]})
    with pytest.raises(TilefishPlanError, match="non-empty list of positive ints"):
        tilefish.plan(pm.block, buckets={"M": []})
    with pytest.raises(TilefishPlanError, match="a list of sizes"):
        tilefish.plan(pm.block, buckets={"M": 64})


# -- graphs that are not the block -----------------------------------------------------------------------------------------------------


R = Dim("R")
from plan_model import Q, W, accumulate, bump, plus, twice  # noqa: E402  (the small in-place graph of the model module)


def test_an_in_place_call_keeps_the_storage_of_the_version_it_consumes():
    p = tilefish.plan(accumulate)
    roles = {b["name"]: b["role"] for b in p.buffers}
    assert roles == {"a": "input", "state": "inout", "t0": "intermediate", "out0": "output"}
    steps = [(s["entry"], [(a["name"], a["binding"]) for a in s["args"]]) for s in p.steps(Q=16)]
    assert steps == [
        ("bump", [("state", "state"), ("y", "a")]),
        ("bump", [("state", "state"), ("y", "a")]),  # the second update is of the same tensor
        ("twice", [("x", "a"), ("out", "t0")]),
        ("bump", [("state", "t0"), ("y", "a")]),  # an intermediate is updated in place too: no second buffer
        ("plus", [("x", "t0"), ("y", "state"), ("out", "out0")]),  # and reads see the newest version, in the order of the IR
    ]


def test_a_graph_without_a_dimension_has_one_bucket():
    @tilefish.graph
    def fixed(a: Tensor[4, W, f32], b: Tensor[4, W, f32]) -> Tensor[4, W, f32]:
        return _fixed_plus(a, b)

    p = tilefish.plan(fixed)
    assert p.dims == {} and p.buckets() == [{}] and [s["entry"] for s in p.steps()] == ["fixed_plus"]


@tilefish.kernel
def fixed_plus(x: Tensor[4, W, f32], y: Tensor[4, W, f32]) -> Tensor[4, W, f32]:
    return x + y


_fixed_plus = fixed_plus


def test_a_dimension_without_buckets_needs_them_from_the_caller():
    @tilefish.kernel
    def free_plus(x: Tensor[R, W, f32], y: Tensor[R, W, f32]) -> Tensor[R, W, f32]:
        return x + y

    @tilefish.graph
    def free(a: Tensor[R, W, f32], b: Tensor[R, W, f32]) -> Tensor[R, W, f32]:
        return free_plus(a, b)

    with pytest.raises(TilefishPlanError, match=r"dimension R has no buckets"):
        tilefish.plan(free)
    p = tilefish.plan(free, buckets={"R": [8, 24]})
    assert p.dims == {"R": (8, 24)} and p.steps(R=24)[0]["args"][0] == {"name": "x", "binding": "a"}
    assert {b["name"]: b["bytes"] for b in p.buffers}["a"] == 24 * 128 * 4  # sized for the largest


def test_a_requirement_of_the_graph_is_checked_at_every_bucket():
    T = Dim("T", divisible_by=8)

    @tilefish.kernel
    def ek(x: Tensor[T, f32]) -> Tensor[T, f32]:
        return x * 2.0

    @tilefish.graph
    def divisible(a: Tensor[T, f32]) -> Tensor[T, f32]:
        return ek(a)

    with pytest.raises(TilefishPlanError, match="divisible_by=8"):
        tilefish.plan(divisible, buckets={"T": [8, 12]})
    assert tilefish.plan(divisible, buckets={"T": [8, 16]}).dims == {"T": (8, 16)}


# -- what a plan refuses -----------------------------------------------------------------------------------------------------------------


def test_a_bucket_an_operator_has_no_binary_for_is_refused_by_the_graphs_own_requirement():
    G = Dim("G", buckets=[32, 64, 128])

    @tilefish.library("gemm_g", acc=f32, impl="cublaslt.gemm")
    def gemm_g(a: Tensor[Dim("G", buckets=[32, 64]), 256, bf16], b: Tensor[256, 256, bf16]) -> Tensor[Dim("G", buckets=[32, 64]), 256, bf16]: ...

    @tilefish.graph
    def wide(a: Tensor[G, 256, bf16], b: Tensor[256, 256, bf16]) -> Tensor[G, 256, bf16]:
        return gemm_g(a, b)

    with pytest.raises(TilefishPlanError, match=r"G in \[32, 64\].*it is 128"):
        tilefish.plan(wide)
    assert tilefish.plan(wide, buckets={"G": [32, 64]}).dims == {"G": (32, 64)}  # the subset the operator has binaries for


def test_a_kernel_or_anything_that_is_not_a_graph_is_not_planned():
    with pytest.raises(TilefishPlanError, match="takes a @tilefish.graph"):
        tilefish.plan(twice)
    with pytest.raises(TilefishPlanError, match="a kernel is launched by tilefish.run directly"):
        tilefish.plan(twice)
    with pytest.raises(TilefishPlanError, match="takes a @tilefish.graph"):
        tilefish.plan(pm.gemm_o)
    with pytest.raises(TilefishPlanError, match="reuse= is True or False"):
        tilefish.plan(pm.block, reuse=1)


def test_arithmetic_in_the_graph_body_has_no_kernel_and_is_refused_with_the_rule():
    @tilefish.graph
    def arith(a: Tensor[Q, W, f32]) -> Tensor[Q, W, f32]:
        return a + a

    with pytest.raises(TilefishPlanError, match=r"arithmetic traced in the graph body.*@tilefish.kernel"):
        tilefish.plan(arith)

    @tilefish.graph
    def cast(a: Tensor[Q, W, f32]) -> Tensor[Q, W, bf16]:
        return twice(a).to(bf16)

    with pytest.raises(TilefishPlanError, match="the op 'cast'"):
        tilefish.plan(cast)


def test_a_store_an_out_parameter_a_runtime_scalar_and_a_nested_graph_are_refused():
    @tilefish.graph
    def stores(a: Tensor[Q, W, f32], y: Out[Tensor[Q, W, f32]]):
        y[...] = twice(a)

    with pytest.raises(TilefishPlanError, match="Out parameter 'y' is not planned"):
        tilefish.plan(stores)

    @tilefish.graph
    def inner(a: Tensor[Q, W, f32]) -> Tensor[Q, W, f32]:
        return twice(a)

    @tilefish.graph
    def outer(a: Tensor[Q, W, f32]) -> Tensor[Q, W, f32]:
        return inner(a)

    with pytest.raises(TilefishPlanError, match="calls another graph"):
        tilefish.plan(outer)

    @tilefish.graph
    def scalar(a: Tensor[Q, W, f32], n: Q) -> Tensor[Q, W, f32]:
        return twice(a)

    with pytest.raises(TilefishPlanError, match="runtime scalar parameter 'n' is not planned"):
        tilefish.plan(scalar)


def test_an_in_place_call_on_a_view_is_refused_because_the_ir_says_the_parent_is_unchanged():
    # a chunk along the first axis of a dense tensor is a contiguous tensor (the types let it into an in-place call); it is still a view of its parent
    @tilefish.kernel
    def bump_half(state: InOut[Tensor[4, 64, f32]], y: Tensor[4, 64, f32]):
        state[...] = state + y

    @tilefish.kernel
    def keep(x: Tensor[8, 64, f32]) -> Tensor[8, 64, f32]:
        return x * 1.0

    @tilefish.kernel
    def mk(x: Tensor[8, 64, f32]) -> Tensor[8, 64, f32]:
        return x * 2.0

    @tilefish.graph
    def on_view(a: Tensor[8, 64, f32]) -> Tensor[8, 64, f32]:
        t = mk(a)
        top, bottom = t.split(4, axis=0)
        bump_half(top, bottom)
        return keep(t)

    with pytest.raises(TilefishPlanError, match=r"updates 'state' in place, but the argument is a view of another tensor"):
        tilefish.plan(on_view)

    # a strided view never gets that far: the types refuse it at the call
    @tilefish.kernel
    def bump_cols(state: InOut[Tensor[8, 32, f32]], y: Tensor[8, 32, f32]):
        state[...] = state + y

    with pytest.raises(tilefish.TilefishTypeError, match="strided view"):

        @tilefish.graph
        def on_strided(a: Tensor[8, 64, f32]) -> Tensor[8, 64, f32]:
            t = mk(a)
            left, right = t.split(32, axis=-1)
            bump_cols(left, right)
            return keep(t)


def test_a_view_that_is_read_after_its_tensor_was_updated_in_place_is_refused_because_it_would_read_the_new_bytes():
    # the IR gives a view the contents its tensor had when the view was made; a plan shares the storage, so it must not read it after an in-place update
    @tilefish.kernel
    def bump8(state: InOut[Tensor[8, 64, f32]], y: Tensor[8, 64, f32]):
        state[...] = state + y

    @tilefish.kernel
    def mk8(x: Tensor[8, 64, f32]) -> Tensor[8, 64, f32]:
        return x * 2.0

    @tilefish.kernel
    def read4(x: Tensor[4, 64, f32]) -> Tensor[4, 64, f32]:
        return x * 3.0

    @tilefish.graph
    def stale(a: Tensor[8, 64, f32]) -> Tensor[4, 64, f32]:
        t = mk8(a)
        top, _ = t.split(4, axis=0)
        bump8(t, a)
        return read4(top)

    with pytest.raises(TilefishPlanError, match=r"reads 'x' = %1 .*a view of a tensor that was updated in place after the view was made"):
        tilefish.plan(stale)

    @tilefish.graph
    def read_before(a: Tensor[8, 64, f32]) -> tuple[Tensor[4, 64, f32], Tensor[8, 64, f32]]:
        t = mk8(a)
        top, _ = t.split(4, axis=0)
        early = read4(top)  # read while the view still means what it meant
        bump8(t, a)
        return early, t.live

    steps = [(s["entry"], [(a["name"], a["binding"]) for a in s["args"]]) for s in tilefish.plan(read_before).steps()]
    assert [s[0] for s in steps] == ["mk8", "read4", "bump8"]

    @tilefish.graph
    def view_of_the_new_version(a: Tensor[8, 64, f32]) -> Tensor[4, 64, f32]:
        t = mk8(a)
        bump8(t, a)
        top, _ = t.live.split(4, axis=0)  # a view of the updated version
        return read4(top)

    assert [s["entry"] for s in tilefish.plan(view_of_the_new_version).steps()] == ["mk8", "bump8", "read4"]


def test_a_view_of_a_graph_parameter_that_is_updated_in_place_is_stale_for_the_same_reason():
    @tilefish.kernel
    def bump_rows(state: InOut[Tensor[8, 64, f32]], y: Tensor[8, 64, f32]):
        state[...] = state + y

    @tilefish.kernel
    def read_rows(x: Tensor[4, 64, f32]) -> Tensor[4, 64, f32]:
        return x * 3.0

    @tilefish.graph
    def on_param(a: Tensor[8, 64, f32], state: InOut[Tensor[8, 64, f32]]) -> Tensor[4, 64, f32]:
        top, _ = state.split(4, axis=0)
        bump_rows(state, a)
        return read_rows(top)

    with pytest.raises(TilefishPlanError, match="a view of a tensor that was updated in place after the view was made"):
        tilefish.plan(on_param)


def test_a_returned_value_must_be_a_fresh_result_not_an_input_a_view_or_a_repeat():
    @tilefish.graph
    def returns_input(a: Tensor[Q, W, f32]) -> Tensor[Q, W, f32]:
        twice(a)
        return a

    with pytest.raises(TilefishPlanError, match="returned value %a .* is the input 'a'"):
        tilefish.plan(returns_input)

    @tilefish.graph
    def returns_view(a: Tensor[Q, W, f32]):
        left, right = twice(a).split(64, axis=-1)
        return left

    with pytest.raises(TilefishPlanError, match="is a view"):
        tilefish.plan(returns_view)

    @tilefish.graph
    def twice_returned(a: Tensor[Q, W, f32]) -> tuple[Tensor[Q, W, f32], Tensor[Q, W, f32]]:
        t = twice(a)
        return t, t

    with pytest.raises(TilefishPlanError, match="returned twice"):
        tilefish.plan(twice_returned)


def test_two_returned_values_are_named_in_the_order_the_graph_returns_them():
    @tilefish.graph
    def swapped(a: Tensor[Q, W, f32]) -> tuple[Tensor[Q, W, f32], Tensor[Q, W, f32]]:
        first = twice(a)
        second = plus(first, a)
        return second, first

    p = tilefish.plan(swapped)
    assert p.outputs == ["out0", "out1"]
    first, last = p.steps(Q=16)
    assert [(a["name"], a["binding"]) for a in first["args"]] == [("x", "a"), ("out", "out1")]  # the first call made the value that is returned second
    assert [(a["name"], a["binding"]) for a in last["args"]] == [("x", "out1"), ("y", "a"), ("out", "out0")]


def test_two_operators_with_one_name_would_share_a_manifest_entry_and_are_refused():
    @tilefish.kernel(name="same")
    def one(x: Tensor[Q, W, f32]) -> Tensor[Q, W, f32]:
        return x * 2.0

    @tilefish.kernel(name="same")
    def other(x: Tensor[Q, W, f32]) -> Tensor[Q, W, f32]:
        return x * 3.0

    @tilefish.graph
    def clash(a: Tensor[Q, W, f32]) -> Tensor[Q, W, f32]:
        return other(one(a))

    with pytest.raises(TilefishPlanError, match="two different operators are both named 'same'"):
        tilefish.plan(clash)


def test_a_library_op_without_an_implementation_has_nothing_to_launch():
    @tilefish.library("opaque", acc=f32)
    def opaque(a: Tensor[Q, W, f32]) -> Tensor[Q, W, f32]: ...

    @tilefish.graph
    def uses(a: Tensor[Q, W, f32]) -> Tensor[Q, W, f32]:
        return opaque(a)

    with pytest.raises(TilefishPlanError, match="has no implementation"):
        tilefish.plan(uses)


def test_the_parameter_names_that_the_plan_makes_up_do_not_collide_with_the_graphs():
    @tilefish.graph
    def names(t0: Tensor[Q, W, f32], out0: Tensor[Q, W, f32], workspace: Tensor[Q, W, f32]) -> Tensor[Q, W, f32]:
        return plus(twice(t0), plus(out0, workspace))

    p = tilefish.plan(names)
    listed = [b["name"] for b in p.buffers]
    assert len(set(listed)) == len(listed) and listed[:3] == ["t0", "out0", "workspace"]
    assert p.outputs == ["out0_"] and "t0_" in listed


def test_a_plan_describes_itself():
    text = tilefish.plan(pm.block, reuse=True).describe()
    assert "plan of graph 'block'" in text and "arena: 9240832 bytes (with reuse); without reuse 9634048, with reuse 9240832" in text
    assert "workspace    workspace    u8" in text and "#2 attention (kernel): t1[view], t1[view], t1[view], t2, n" in text
    assert text.splitlines()[-1].startswith("  #8 add (kernel)")


def test_the_plan_of_a_graph_is_made_without_torch_and_names_every_binary_by_its_stem():
    p = tilefish.plan(pm.block)
    assert sorted(p.callees) == sorted(
        {"rmsnorm", "attention", "add_rmsnorm", "gelu", "add", *(f"{op}.M{m}" for op in ("gemm_qkv", "gemm_o", "gemm_up", "gemm_down") for m in pm.BUCKETS)}
    )
    assert json.loads(p.to_json()) == p.doc
