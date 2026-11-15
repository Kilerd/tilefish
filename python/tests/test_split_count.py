"""A ``split`` whose number of chunks depends on a dimension: ``x.split(16, axis=1)`` of ``x: [M, C]`` with ``C = Dim("C", divisible_by=16)``.

A trace cannot return ``C // 16`` Python values, so the split is a :class:`~tilefish.ops.Chunks`, a sequence of symbolic length: ``chunks[0]`` and
``chunks[-1]`` always exist, ``chunks[k]`` where the dimension's smallest size proves it, each a view with the strides of ``x`` and the offset of
its start (exactly a chunk of a split whose count is a number); ``len()``, iteration and unpacking raise. A count that is a number gives the
tuple, as it always did. The bytes of a plan that reads such chunks: ``test_gpu_split_count.py``.
"""

import pytest

import tilefish
from tilefish import BlockScaled, Dim, Tensor, TilefishLoweringError, TilefishTypeError, bf16, e4m3, f32, ue8m0
from tilefish.ops import Chunks

M = Dim("M")
C = Dim("C", divisible_by=16)
C64 = Dim("C64", divisible_by=64)
Cb = Dim("Cb", buckets=[64, 128])
H = Dim("H")


def layout(v):
    t = v.type
    return [str(s) if s is not None else None for s in t.layout_strides()], None if t.offset is None else str(t.offset), t.contiguous


def test_a_count_that_depends_on_a_dimension_is_a_sequence_of_chunks(tr):
    x = tr.val(Tensor[M, C, bf16])
    chunks = x.split(16, axis=1)
    assert isinstance(chunks, Chunks) and not tr.nodes  # nothing is traced until a chunk is taken
    assert str(chunks.count) == "cdiv(C, 16)" and chunks.known == 1 and chunks.size == 16 and chunks.axis == 1
    first, last = chunks[0], chunks[-1]
    assert first.type == Tensor[M, 16, bf16].as_view() and last.type == first.type
    assert layout(first) == (["C", "1"], "0", False) and layout(last) == (["C", "1"], "C - 16", False)  # the strides of x, the chunk's start
    assert [n.op for n in tr.nodes] == ["split", "split"]
    assert tr.nodes[0].attrs == {"axis": 1, "sizes": (16,), "start": 0} and tr.nodes[1].attrs["start"] == C - 16
    assert repr(chunks) == "<Chunks: the cdiv(C, 16) chunks of size 16 of axis 1 of %p0 (Tensor[M, C, bf16])>"
    assert bool(chunks)  # at least one chunk


def test_a_static_count_stays_a_tuple(tr):
    assert isinstance(tr.val(Tensor[M, 48, bf16]).split(16, axis=1), tuple)
    q, k, v = tr.val(Tensor[M, 3 * H, bf16]).split(H, axis=1)  # 3 * H in chunks of H: 3, whatever H is
    assert layout(k) == (["3*H", "1"], "H", False)
    assert tr.nodes[-1].attrs == {"axis": 1, "sizes": (H, H, H)}  # no start: the node of before
    assert len(tr.val(Tensor[M, C, bf16]).split([16, C - 16], axis=1)) == 2  # sizes that add up: as before


def test_the_chunks_that_exist_are_those_the_smallest_size_proves(tr):
    wide = tr.val(Tensor[M, C64, f32]).split(16, axis=1)
    assert wide.known == 4
    assert layout(wide[3]) == (["C64", "1"], "48", False) and layout(wide[-4]) == (["C64", "1"], "C64 - 64", False)
    for k in (4, -5):
        with pytest.raises(TilefishTypeError, match=rf"split: chunk {k} of the cdiv\(C64, 16\) chunks .* is not provably there: C64 can be as small as 64, which has 4 chunk\(s\) of 16, so chunks\[0\] to chunks\[3\] and chunks\[-1\] to chunks\[-4\] exist"):
            wide[k]
    bucketed = tr.val(Tensor[M, Cb, f32]).split(32, axis=1)  # every bucket is a multiple of 32, the smallest has 2 chunks
    assert isinstance(bucketed, Chunks) and bucketed.known == 2 and layout(bucketed[1]) == (["Cb", "1"], "32", False)
    with pytest.raises(TilefishTypeError, match="not provably there"):
        bucketed[2]
    narrow = tr.val(Tensor[M, C, f32]).split(16, axis=1)
    with pytest.raises(TilefishTypeError, match=r"chunks\[0\] to chunks\[0\] and chunks\[-1\] to chunks\[-1\] exist"):
        narrow[1]


def test_the_count_cannot_be_used(tr):
    chunks = tr.val(Tensor[M, C, bf16]).split(16, axis=1)
    count = r"their number depends on C, which a trace does not know. Take a chunk with chunks\[k\] \(chunks\[0\] and chunks\[-1\] always exist\)"
    with pytest.raises(TilefishTypeError, match="split: len\\(\\) of the cdiv\\(C, 16\\) chunks .*: " + count):
        len(chunks)
    with pytest.raises(TilefishTypeError, match="cannot iterate over or unpack .*" + count):
        a, b = chunks
    with pytest.raises(TilefishTypeError, match="cannot iterate over or unpack"):
        list(chunks)
    with pytest.raises(TilefishTypeError, match="cannot iterate over or unpack"):
        for _ in chunks:
            pass
    with pytest.raises(TilefishTypeError, match="are taken one by one .*; a slice of them has no static length"):
        chunks[0:2]
    for bad in ("0", 1.0, True, M):
        with pytest.raises(TilefishTypeError, match="a chunk is taken with a python int"):
            chunks[bad]
    assert not tr.nodes


def test_a_count_needs_a_size_that_provably_divides_the_dimension(tr):
    with pytest.raises(TilefishTypeError, match=r"has size H, which is not a whole number of chunks of size 16 \(a count that depends on a dimension needs H provably divisible by 16: Dim\(..., divisible_by=16\)"):
        tr.val(Tensor[M, H, bf16]).split(16, axis=1)
    with pytest.raises(TilefishTypeError, match="not a whole number of chunks of size 32"):
        tr.val(Tensor[M, C, bf16]).split(32, axis=1)  # 16 divides C, 32 does not
    with pytest.raises(TilefishTypeError, match="not a whole number of chunks of size 48"):
        tr.val(Tensor[M, Dim("Cs", buckets=[96, 128]), bf16]).split(48, axis=1)  # 128 is not a multiple of 48
    with pytest.raises(TilefishTypeError, match=r"not a whole number of chunks of size H$"):
        tr.val(Tensor[M, C, bf16]).split(H, axis=1)  # a symbolic size: no count at all
    with pytest.raises(TilefishTypeError, match=r"not a whole number of chunks of size 16$"):
        tr.val(Tensor[M, 40, bf16]).split(16, axis=1)  # a static dimension: as before


def test_a_requirement_of_the_trace_proves_the_divisibility_too(tr):
    x = tr.val(Tensor[M, H, bf16])
    tr.ctx.require_divisible(H, 16, "a call")
    assert isinstance(x.split(16, axis=1), Chunks)


def test_chunks_along_the_leading_axis_are_dense_slabs(tr):
    y = tr.val(Tensor[C, 64, f32])
    chunks = y.split(16, axis=0)
    assert layout(chunks[0]) == (["64", "1"], "0", True) and chunks[0].type == Tensor[16, 64, f32]
    assert layout(chunks[-1]) == (["64", "1"], "64*C - 1024", True)  # 16-byte aligned: C is a multiple of 16
    odd = tr.val(Tensor[Dim("E", divisible_by=2), 3, bf16]).split(2, axis=0)
    assert odd[0].type.contiguous and not odd[-1].type.contiguous  # (E - 2) * 3 bf16 elements in: not provably on 16 bytes


def test_a_block_scaled_tensor(tr):
    mx = BlockScaled(e4m3, ue8m0, 32)
    x = tr.val(Tensor[C64, 64, mx])
    chunks = x.split(16, axis=0)
    assert chunks[0].type.contiguous and chunks[-1].type == Tensor[16, 64, mx]  # data and scales on 16 bytes: (C64 - 16) * 64 bytes in


def test_the_chunks_are_of_the_tensor_as_it_was_split(tr):
    x = tr.val(Tensor[M, C, f32], name="x", access="inout")
    chunks = x.split(16, axis=1)
    chunks[0]
    x[...] = x * 2.0
    with pytest.raises(TilefishTypeError, match="which the store into 'x' already consumed"):
        chunks[-1]


def test_a_kernel_with_such_a_chunk_is_not_lowered():
    @tilefish.kernel
    def head(x: Tensor[M, C, f32]) -> Tensor[M, 16, f32]:
        return x.split(16, axis=1)[0] + 0.0

    with pytest.raises(TilefishLoweringError) as e:
        head.lower()
    assert e.value.rule == "split"


def test_in_a_graph():
    Mb = Dim("Mb", buckets=[8])

    @tilefish.graph
    def ends(x: Tensor[Mb, Cb, bf16]):
        chunks = x.split(32, axis=1)
        return tilefish.contiguous(chunks[0]), tilefish.contiguous(chunks[-1])

    first, *_, last = (n for n in ends.graph.nodes if n.op == "split")
    assert first.attrs["start"] == 0 and last.attrs["start"] == Cb - 32
    assert [n.attrs.get("kind") for n in ends.graph.nodes] == [None, "extern", None, "extern"]


def test_value_split_documents_the_sequence():
    assert "Chunks" in tilefish.trace.Value.split.__doc__
    with pytest.raises(TilefishTypeError, match="needs a size and an axis"):

        @tilefish.kernel
        def k(x: Tensor[M, C, f32]) -> Tensor[M, C, f32]:
            x.split(16)
            return x
