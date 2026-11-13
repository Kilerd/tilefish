"""Strides that several views share: ``StrideOf(("q", "k", "v"), axis)``.

A kernel that reads several views of one buffer with **one** stride pair (the q, k and v column blocks of a fused ``[B, 3 * D]`` tensor, at offsets
0, ``D`` and ``2 * D``) declares one scalar per shared stride, ``s_row: StrideOf(("q", "k", "v"), 0)``. Each view is its own ``Strided[...]`` pointer
with its own offset; the views share the stride *dimension*, so a call binds it from the first one and holds every other one to it: a trace call
refuses a view that has another stride, and so does a launch (kind ``stride``), in Python and in Rust, before anything runs. The compile spec, the
manifest entry and the declaration digest write the scalar's ``stride_of`` as a list of the views it carries the stride of; a stride of one view is
written as it always was. Nothing here needs a GPU (``test_gpu_shared_strides.py`` runs the kernels).
"""

import copy
import json

import pytest
from built_fakes import FAKE_SOURCE_DIGEST, fake_build
from fake_torch import FakeTensor, prepare, verdict

import tilefish
from tilefish import (
    Dim,
    Manifest,
    Out,
    StrideOf,
    Strided,
    Tensor,
    TilefishDeclarationError,
    TilefishManifestError,
    TilefishPreconditionError,
    TilefishSpecError,
    TilefishTypeError,
    bf16,
    const,
)
from tilefish.aot import CompileSpec, compile_specs, declaration_digest, write_manifest
from tilefish.hosts import rust

R = Dim("R")
B = Dim("B", buckets=[8, 24])


def qkv_op(name="qkv_sum", align=None, divisible_by=None):
    """``o = q + k + v`` of three ``[R, 4, 64]`` views read with one stride pair (the extern of ``gpu_kernels_p4.py``, by its module name)."""
    view = Strided[Tensor[R, 4, 64, bf16], 16] if align else Strided[Tensor[R, 4, 64, bf16]]

    @tilefish.extern(triton="gpu_kernels_p4:qkv_sum_kernel", num_warps=1, grid=lambda R: (R, 4), name=name)
    def op(
        q: view,
        k: view,
        v: view,
        o: Out[Tensor[R, 4, 64, bf16]],
        s_row: StrideOf(("q", "k", "v"), 0, divisible_by=divisible_by),
        s_head: StrideOf(("q", "k", "v"), 1, divisible_by=divisible_by),
        D: const = 64,
    ): ...

    return op


qkv_sum = qkv_op()
qkv_sum16 = qkv_op("qkv_sum16", align=16, divisible_by=8)


# -- the declaration ----------------------------------------------------------------------------------------------------------------------------


def test_the_views_share_the_stride_dimension_and_keep_their_own_offsets():
    q, k, v = qkv_sum.signature.params[:3]
    s_row, s_head = qkv_sum.signature.params[4:6]
    assert q.type.strides == k.type.strides == v.type.strides == (s_row.dim, s_head.dim, 1)
    assert [p.type.offset.name for p in (q, k, v)] == ["q_offset", "k_offset", "v_offset"]
    assert s_row.stride_of == (("q", 0), ("k", 0), ("v", 0)) and s_head.stride_of == (("q", 1), ("k", 1), ("v", 1))
    assert str(s_row) == "s_row: StrideOf(('q', 'k', 'v'), 0)"


def test_the_annotation_takes_a_name_or_a_tuple_of_names():
    assert StrideOf(("x",), 0) == StrideOf("x", 0) and StrideOf(("x",), 0).arg == "x"  # one view is one view
    assert StrideOf(["q", "k"], 1).arg == ("q", "k") and StrideOf(["q", "k"], 1).targets == (("q", 1), ("k", 1))
    for bad, message in (
        ((), "names a tensor parameter"),
        (("q", "q"), "named twice"),
        (("q", "not an identifier"), "names a tensor parameter"),
        (("q", 3), "names a tensor parameter"),
        (5, "names a tensor parameter"),
    ):
        with pytest.raises(TilefishDeclarationError, match=message):
            StrideOf(bad, 0)


def test_a_declaration_whose_shared_stride_names_something_else_is_refused():
    with pytest.raises(TilefishDeclarationError, match=r"s0 is StrideOf\(\('x', 'y'\), 0\), but 'y' is not a Strided\[\.\.\.\] tensor parameter"):

        @tilefish.extern(triton="m:f", grid=(1,))
        def f1(x: Strided[Tensor[4, 8, bf16]], y: Tensor[4, 8, bf16], s0: StrideOf(("x", "y"), 0)): ...

    with pytest.raises(TilefishDeclarationError, match=r"s1 is the stride of axis 1 of 'y', which has rank 1"):

        @tilefish.extern(triton="m:f", grid=(1,))
        def f2(x: Strided[Tensor[4, 8, bf16]], y: Strided[Tensor[8, bf16]], s0: StrideOf("x", 0), s1: StrideOf(("x", "y"), 1)): ...

    with pytest.raises(TilefishDeclarationError, match=r"the stride of axis 0 of 'y' is declared twice \(s0 and s1\)"):

        @tilefish.extern(triton="m:f", grid=(1,))
        def f3(x: Strided[Tensor[4, 8, bf16]], y: Strided[Tensor[4, 8, bf16]], s0: StrideOf(("x", "y"), 0), s1: StrideOf("y", 0)): ...


# -- the compile spec, the manifest entry, the digest --------------------------------------------------------------------------------------------


def test_the_spec_writes_the_views_of_a_shared_stride_as_a_list():
    (spec,) = compile_specs(qkv_sum)
    s_row, s_head = spec.args[4:6]
    assert s_row.to_dict()["stride_of"] == [{"arg": "q", "axis": 0}, {"arg": "k", "axis": 0}, {"arg": "v", "axis": 0}]
    assert s_head.stride_of == (("q", 1), ("k", 1), ("v", 1))
    assert [a.strides for a in spec.args[:3]] == [("s_row", "s_head", 1)] * 3
    assert [a.offset for a in spec.args[:3]] == ["q_offset", "k_offset", "v_offset"]
    assert spec.dims == {"R": {}, "k_offset": {}, "q_offset": {}, "s_head": {}, "s_row": {}, "v_offset": {}}  # one name per shared stride
    back = CompileSpec.from_json(spec.to_json())
    back.verify_against(qkv_sum)
    assert back.verified and back.args[4].stride_of == (("q", 0), ("k", 0), ("v", 0))


def test_a_16_byte_aligned_shared_stride_is_a_divisibility_hint_on_each_pointer():
    (spec,) = compile_specs(qkv_sum16)
    assert spec.attrs == {(i,): [["tt.divisibility", 16]] for i in (0, 1, 2, 3)}  # the three views and the output
    assert {p.arg: p.bytes for p in spec.preconditions if p.kind == "pointer_alignment"} == {"q": 16, "k": 16, "v": 16, "o": 16}
    # a stride that is a multiple of 8 elements is a constraint of its dimension (16 bytes of bf16), checked once for the three views
    assert spec.dims["s_row"] == spec.dims["s_head"] == {"divisible_by": 8} and spec.dims["q_offset"] == {"divisible_by": 8}


def test_the_digest_of_a_stride_of_one_view_is_unchanged_and_a_shared_one_differs():
    @tilefish.extern(triton="m:f", grid=(1,), name="one")
    def one(x: Strided[Tensor[4, 8, bf16]], y: Strided[Tensor[4, 8, bf16]], s: StrideOf("x", 0), t: StrideOf("y", 0)): ...

    @tilefish.extern(triton="m:f", grid=(1,), name="one")
    def shared(x: Strided[Tensor[4, 8, bf16]], y: Strided[Tensor[4, 8, bf16]], s: StrideOf(("x", "y"), 0), t: const = 0): ...

    assert declaration_digest(one) != declaration_digest(shared)
    (spec,) = compile_specs(one)
    assert spec.args[2].to_dict()["stride_of"] == {"arg": "x", "axis": 0}  # one view: the object, as it always was


def rejects(change, message, op=qkv_sum):
    doc = copy.deepcopy(compile_specs(op)[0].to_dict())
    change(doc)
    with pytest.raises(TilefishSpecError, match=message):
        CompileSpec.from_dict(doc)


def test_a_serialized_shared_stride_is_checked():
    s_row = 4
    rejects(lambda d: d["args"][s_row].update(stride_of=[{"arg": "q", "axis": 0}]), "two or more of them")
    rejects(lambda d: d["args"][s_row].update(stride_of=[{"arg": "q", "axis": 0}, {"arg": "q", "axis": 0}]), "names axis 0 of 'q' twice")
    rejects(lambda d: d["args"][s_row].update(stride_of=[{"arg": "q", "axis": 0}, "k"]), "must be an object")
    rejects(lambda d: d["args"][s_row].update(stride_of=[{"arg": "q", "axis": 0}, {"arg": "k"}]), "is missing")
    rejects(lambda d: d["args"][s_row].update(stride_of=[{"arg": "q", "axis": 0}, {"arg": "o", "axis": 0}]), "which is not a strided pointer argument")
    rejects(lambda d: d["args"][s_row].update(stride_of=[{"arg": "q", "axis": 0}, {"arg": "k", "axis": 1}]), "carries 's_row' as the stride of axis 1 of 'k'")
    rejects(lambda d: d["args"][s_row].update(stride_of="q"), "must be an object")
    # a view whose stride entry no scalar carries any more
    rejects(lambda d: d["args"][s_row].update(stride_of=[{"arg": "q", "axis": 0}, {"arg": "k", "axis": 0}]), "the stride 's_row' of axis 0 of 'v' is carried by no scalar")


def test_the_manifest_entry_has_the_list_and_both_readers_take_it(tmp_path):
    (spec,) = compile_specs(qkv_sum)
    write_manifest([fake_build(spec)], tmp_path / "manifest.json", cubins=True)
    entry = json.loads((tmp_path / "manifest.json").read_text())["kernels"][0]
    params = {p["name"]: p for p in entry["abi"]["params"]}
    assert params["s_row"]["stride_of"] == [{"arg": "q", "axis": 0}, {"arg": "k", "axis": 0}, {"arg": "v", "axis": 0}]
    assert [params[n]["strides"] for n in "qkv"] == [["s_row", "s_head", 1]] * 3
    m = Manifest.load(tmp_path)
    assert m.built_kernel(spec, "sm_90", ir_digest=FAKE_SOURCE_DIGEST, ir_triton="3.5.1").stem == "qkv_sum"
    source = rust.render(m.check_pin(m.digest))  # the emitter reads it too, and documents the one stride
    assert "the stride of axis 0 of `q`, `k` and `v`, in elements (one stride, which each of them must have)" in source


@pytest.mark.parametrize(
    "stride_of, message",
    [
        ([{"arg": "q", "axis": 0}], "two or more of them"),
        ([{"arg": "q", "axis": 0}, {"arg": "q", "axis": 0}], "each once"),
        ([{"arg": "q", "axis": 0}, ["k", 0]], "stride_of must be"),
        ("q", "stride_of must be an object or a list"),
    ],
)
def test_the_manifest_reader_refuses_a_malformed_shared_stride(tmp_path, stride_of, message):
    (spec,) = compile_specs(qkv_sum)
    write_manifest([fake_build(spec)], tmp_path / "manifest.json", cubins=True)
    path = tmp_path / "manifest.json"
    doc = json.loads(path.read_text())
    next(p for p in doc["kernels"][0]["abi"]["params"] if p["name"] == "s_row")["stride_of"] = stride_of
    path.write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match=message):
        Manifest.load(tmp_path)


# -- calls in a trace ---------------------------------------------------------------------------------------------------------------------------


def test_the_q_k_and_v_of_a_fused_tensor_bind_the_shared_strides_once():
    @tilefish.graph
    def g(qkv: Tensor[B, 768, bf16]):
        q, k, v = (qkv.reshape(B, 3, 4, 64)[:, i] for i in range(3))
        return qkv_sum(q, k, v)

    assert str(g.graph.outputs[0].type) == "Tensor[B, 4, 64, bf16]"


def test_a_view_with_another_stride_is_refused_at_the_trace_call():
    with pytest.raises(
        TilefishTypeError,
        match=r"qkv_sum: argument 'k' \(Tensor\[B, 4, 64, bf16\]\): the stride of axis 0 is 256, but the signature requires s_row = 768 "
        r"\(the kernel takes one stride for the views that share it: each must have it\) \(s_row was bound to 768 by argument 'q'\)",
    ):

        @tilefish.graph
        def g(qkv: Tensor[B, 3, 4, 64, bf16], other: Tensor[B, 4, 64, bf16]):
            return qkv_sum(qkv[:, 0], other, qkv[:, 2])


def test_views_of_three_tensors_with_the_same_strides_are_fine():
    @tilefish.graph
    def g(a: Tensor[B, 4, 64, bf16], b: Tensor[B, 4, 64, bf16], c: Tensor[B, 4, 64, bf16]):
        return qkv_sum(a, b, c)  # what is shared is the stride, not the buffer

    assert len(g.graph.outputs) == 1


def test_a_plan_binds_the_three_views_in_one_step():
    @tilefish.kernel
    def doubled(x: Tensor[B, 768, bf16]) -> Tensor[B, 768, bf16]:
        return x * 2.0

    @tilefish.graph
    def fused(x: Tensor[B, 768, bf16]):
        qkv = doubled(x).reshape(B, 3, 4, 64)
        return qkv_sum16(qkv[:, 0], qkv[:, 1], qkv[:, 2])

    plan = tilefish.plan(fused)
    (step,) = [s for s in plan.steps(B=24) if s["entry"] == "qkv_sum16"]
    assert [(a["name"], a["binding"], a.get("view")) for a in step["args"]] == [
        ("q", "t0", {"shape": [24, 4, 64], "strides": [768, 64, 1], "offset": 0}),
        ("k", "t0", {"shape": [24, 4, 64], "strides": [768, 64, 1], "offset": 256}),
        ("v", "t0", {"shape": [24, 4, 64], "strides": [768, 64, 1], "offset": 512}),
        ("o", "out0", None),
    ]


# -- a launch (Python's decision, which the Rust host is held to: python_launches.json) ----------------------------------------------------------

SLAB = 1 << 36


def fake_qkv(rows, strides=(768, 64, 1), base=1 << 44):
    """q, k and v views of a fused ``[rows, 3, 4, 64]`` bf16 tensor at ``base``: offsets 0, 256 and 512 elements."""
    return [FakeTensor((rows, 4, 64), "bf16", base + 2 * 256 * i, strides) for i in range(3)]


def test_a_launch_passes_the_shared_strides_once():
    prepared, error = prepare(qkv_sum, fake_qkv(5), next_slab=(1 << 44) + SLAB)
    assert error is None
    assert prepared.scalars == {"s_row": 768, "s_head": 64}
    v = verdict(prepared, None)
    assert v["grid"] == [5, 4, 1] and v["params"][-4:] == [768, 64, 0, 0]


@pytest.mark.parametrize("which", [1, 2])
@pytest.mark.parametrize("strides", [(769, 64, 1), (768, 65, 1), (256, 64, 1)])
def test_a_launch_refuses_a_view_without_the_shared_stride(which, strides):
    views = fake_qkv(5)
    views[which] = FakeTensor((5, 4, 64), "bf16", views[which].data_ptr(), strides)
    prepared, error = prepare(qkv_sum, views, next_slab=(1 << 44) + SLAB)
    assert prepared is None and isinstance(error, TilefishPreconditionError) and error.kind == "stride"
    name = "qkv"[which]
    assert f"argument '{name}'" in str(error) and "each must have it" in str(error)


def test_one_row_has_the_canonical_strides_on_every_view():
    """A view of one row: the stride of its size-1 axis is the one a dense tensor would have, on each view alike."""
    views = [FakeTensor((1, 4, 64), "bf16", (1 << 44) + 512 * i, (1 << 40, 64, 1)) for i in range(3)]
    prepared, error = prepare(qkv_sum, views, next_slab=(1 << 44) + SLAB)
    assert error is None and prepared.scalars == {"s_row": 256, "s_head": 64}
