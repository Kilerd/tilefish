"""A split chunk is a contiguous, aligned tensor only if it provably starts on a 16-byte boundary; otherwise it is a view."""

import pytest

import tilefish
from tilefish import (
    BlockScaled,
    Dim,
    Out,
    Tensor,
    TilefishTypeError,
    bf16,
    e2m1,
    e4m3,
    f32,
    i64,
    u8,
    ue8m0,
)
from tilefish.aot import compile_specs

A = Dim("A")
B = Dim("B")
C = Dim("C")


def flags(chunks):
    return [c.type.contiguous for c in chunks]


# Hand-computed: the chunk after the first starts (elements before it) * itemsize bytes in.
@pytest.mark.parametrize(
    "dtype,shape,sizes,expected",
    [
        # bf16: 2 bytes per element, 8 elements per 16 bytes
        (bf16, (10, 3), [3, 7], [True, False]),  # 9 elements = 18 bytes
        (bf16, (16, 3), [8, 8], [True, True]),  # 24 elements = 48 bytes
        (bf16, (16, 3), [2, 14], [True, False]),  # 6 elements = 12 bytes
        (bf16, (12, 4), [2, 10], [True, True]),  # 8 elements = 16 bytes
        (bf16, (12, 4), [1, 11], [True, False]),  # 4 elements = 8 bytes
        # f32: 4 bytes, 4 elements per 16 bytes
        (f32, (8, 3), [4, 4], [True, True]),  # 12 elements = 48 bytes
        (f32, (8, 3), [1, 7], [True, False]),  # 3 elements = 12 bytes
        (f32, (8, 3), [2, 6], [True, False]),  # 6 elements = 24 bytes
        (f32, (8, 3), [4, 3, 1], [True, True, False]),  # 12 elements = 48 bytes, 21 elements = 84 bytes
        # u8: 1 byte
        (u8, (32, 16), [5, 27], [True, True]),  # 80 bytes
        (u8, (32, 3), [5, 27], [True, False]),  # 15 bytes
        (u8, (32, 3), [16, 16], [True, True]),  # 48 bytes
        # i64: 8 bytes, 2 elements per 16 bytes
        (i64, (4, 1), [1, 3], [True, False]),  # 8 bytes
        (i64, (4, 1), [2, 2], [True, True]),  # 16 bytes
        # sub-byte: e2m1 packs two elements per byte, 32 elements per 16 bytes
        (e2m1, (16, 4), [8, 8], [True, True]),  # 32 elements = 16 bytes
        (e2m1, (16, 4), [4, 12], [True, False]),  # 16 elements = 8 bytes
        (e2m1, (16, 4), [2, 14], [True, False]),  # 8 elements = 4 bytes
        (e2m1, (24, 4), [8, 8, 8], [True, True, True]),  # 16 bytes, 32 bytes
        # 8-bit float scale types
        (ue8m0, (8, 4), [4, 4], [True, True]),  # 16 bytes
        (e4m3, (8, 3), [3, 5], [True, False]),  # 9 bytes
    ],
)
def test_leading_axis_chunks_must_start_on_a_16_byte_boundary(tr, dtype, shape, sizes, expected):
    x = tr.val(Tensor[(*shape, dtype)])
    chunks = x.split(sizes, axis=0)
    assert flags(chunks) == expected
    for chunk, ok in zip(chunks, expected):
        assert chunk.type == (Tensor[(*chunk.shape, dtype)] if ok else Tensor[(*chunk.shape, dtype)].as_view())


def test_the_whole_tensor_is_one_aligned_chunk(tr):
    (whole,) = tr.val(Tensor[16, 3, bf16]).split([16], axis=0)
    assert whole.type.contiguous


def test_a_view_chunk_is_rejected_by_declared_operators_with_the_alignment_hint(tr):
    @tilefish.extern(triton="m:f", grid=lambda A: (A,))
    def ext(x: Tensor[A, 3, bf16], y: Out[Tensor[A, 3, bf16]]): ...

    @tilefish.graph
    def aligned(x: Tensor[16, 3, bf16]):
        a, b = x.split([8, 8], 0)
        return ext(a), ext(b)

    assert [str(v.type) for v in aligned.graph.outputs] == ["Tensor[8, 3, bf16]"] * 2

    with pytest.raises(
        TilefishTypeError,
        match=r"ext: argument 'x' is a strided view \(View\[7, 3, bf16\]\); this parameter takes a contiguous tensor: pass the unsplit tensor, "
        r"compute from the view \(x \+ 0 is a contiguous copy\), or have an extern declare the parameter Strided\[\.\.\.\] "
        r"\(a chunk that does not start on a 16-byte boundary is a view too\)",
    ):

        @tilefish.graph
        def misaligned(x: Tensor[10, 3, bf16]):
            a, b = x.split([3, 7], 0)  # b starts at 3 * 3 * 2 = 18 bytes
            return ext(a), ext(b)

    @tilefish.graph
    def first_is_fine(x: Tensor[10, 3, bf16]):
        a, b = x.split([3, 7], 0)
        return ext(a), ext(b + 0)  # the first chunk is aligned; a view is materialized by computing from it

    assert len(first_is_fine.graph.outputs) == 2
    (spec,) = compile_specs(ext)
    assert spec.attrs[(0,)] == [["tt.divisibility", 16]]  # only ever given to aligned pointers


def test_symbolic_chunks_need_provable_divisibility(tr):
    a, b = Dim("a"), Dim("b")
    x = tr.val(Tensor[a + b, 4, bf16])  # chunk b starts 4a elements in: 8a bytes
    assert flags(x.split([a, b], axis=0)) == [True, False]  # a is not known to be even

    even = Dim("even", divisible_by=2)  # 4 * 2 = 8 elements = 16 bytes
    assert flags(tr.val(Tensor[even + b, 4, bf16]).split([even, b], axis=0)) == [True, True]
    odd_ok = Dim("eight", divisible_by=8)
    assert flags(tr.val(Tensor[3 * odd_ok, 3, bf16]).split(odd_ok, axis=0)) == [True, True, True]  # 24 el per chunk
    assert flags(tr.val(Tensor[3 * a, 3, bf16]).split(a, axis=0)) == [True, False, False]

    # facts recorded by a requirement while tracing count
    known = Dim("known")
    t = tr.val(Tensor[2 * known, 8, f32])
    assert flags(t.split(known, axis=0)) == [True, True]  # 8 f32 per row: 32 bytes per row, any row count
    t2 = tr.val(Tensor[2 * known, 3, f32])
    assert flags(t2.split(known, axis=0)) == [True, False]
    tr.ctx.require_divisible(known, 4, "test")
    assert flags(tr.val(Tensor[2 * known, 3, f32]).split(known, axis=0)) == [True, True]


def test_a_product_of_symbolic_dims_is_never_provably_aligned(tr):
    x = tr.val(Tensor[A + B, C, bf16])
    assert flags(x.split([A, B], axis=0)) == [True, False]  # the offset A*C is not affine
    c8 = Dim("c8", divisible_by=8)
    assert flags(tr.val(Tensor[A + B, c8, bf16]).split([A, B], axis=0)) == [True, False]


def test_chunks_along_a_later_axis_after_size_one_dims(tr):
    x = tr.val(Tensor[1, 16, f32])
    assert flags(x.split([4, 12], axis=1)) == [True, True]  # 16 bytes
    assert flags(x.split([2, 14], axis=1)) == [True, False]  # 8 bytes
    assert flags(tr.val(Tensor[2, 16, f32]).split([4, 12], axis=1)) == [False, False]  # not dense: views
    assert flags(tr.val(Tensor[1, 16, 3, f32]).split([4, 12], axis=1)) == [True, True]  # 4 * 3 elements = 48 bytes
    assert flags(tr.val(Tensor[1, 16, 3, f32]).split([1, 15], axis=1)) == [True, False]  # 12 bytes


def test_chunks_of_a_view_stay_views(tr):
    c, _ = tr.val(Tensor[16, 8, bf16]).split(4, axis=1)
    assert not c.type.contiguous
    assert flags(c.split([8, 8], axis=0)) == [False, False]


def test_block_scaled_chunks_need_an_aligned_scale_tensor_too(tr):
    mx = BlockScaled(e4m3, ue8m0, 32)  # scale tensor [rows, 4] for 128 columns
    x = tr.val(Tensor[64, 128, mx])
    assert flags(x.split([32, 32], axis=0)) == [True, True]  # data 4096 B, scales 128 B
    assert flags(x.split([16, 48], axis=0)) == [True, True]  # data 2048 B, scales 64 B
    assert flags(x.split([1, 63], axis=0)) == [True, False]  # data 128 B but scales 4 B
    assert flags(x.split([4, 60], axis=0)) == [True, True]  # data 512 B, scales 16 B

    packed = BlockScaled(e2m1, ue8m0, 32)  # data packs 2 per byte, scales [rows, 2] for 64 columns
    y = tr.val(Tensor[8, 64, packed])
    assert flags(y.split([3, 5], axis=0)) == [True, False]  # data 96 B (ok) but scales 6 B
    assert flags(y.split([8], axis=0)) == [True]
    assert flags(tr.val(Tensor[16, 64, packed]).split([8, 8], axis=0)) == [True, True]  # scales 16 B


# Splitting along the blocked axis itself: a chunk that starts `before` elements in has its scales at
# element cdiv(before, block) of the scale tensor, not at `before`.
@pytest.mark.parametrize(
    "dtype,shape,axis,sizes,expected",
    [
        # e4m3 data (1 byte) with 1 byte scales, block 32 along the last axis of a single row
        (BlockScaled(e4m3, ue8m0, 32), (1, 1024), 1, [32, 992], [True, False]),  # data 32 B, scales 1 B
        (BlockScaled(e4m3, ue8m0, 32), (1, 1024), 1, [64, 960], [True, False]),  # data 64 B, scales 2 B
        (BlockScaled(e4m3, ue8m0, 32), (1, 1024), 1, [256, 768], [True, False]),  # data 256 B, scales 8 B
        (BlockScaled(e4m3, ue8m0, 32), (1, 1024), 1, [512, 512], [True, True]),  # data 512 B, scales 16 B
        (BlockScaled(e4m3, ue8m0, 32), (1, 1024), 1, [512, 256, 256], [True, True, False]),  # scales 16 B, then 24 B
        # the same on a rank-1 tensor with a non-negative blocked axis
        (BlockScaled(e4m3, ue8m0, 32, axis=0), (1024,), 0, [32, 992], [True, False]),
        (BlockScaled(e4m3, ue8m0, 32, axis=0), (1024,), 0, [512, 512], [True, True]),
        # e2m1 data packs two elements per byte: 32 elements are 16 B of data but still 1 B of scales
        (BlockScaled(e2m1, ue8m0, 32), (1, 1024), 1, [32, 992], [True, False]),
        (BlockScaled(e2m1, ue8m0, 32), (1, 1024), 1, [512, 512], [True, True]),  # data 256 B, scales 16 B
        # a larger block moves the scale offset further down
        (BlockScaled(e4m3, ue8m0, 128), (1, 2048), 1, [512, 1536], [True, False]),  # data 512 B, scales 4 B
        (BlockScaled(e4m3, ue8m0, 128), (1, 2048), 1, [1024, 1024], [True, False]),  # data 1024 B, scales 8 B
    ],
)
def test_splitting_along_the_blocked_axis_aligns_the_scale_tensor_by_blocks_not_elements(tr, dtype, shape, axis, sizes, expected):
    x = tr.val(Tensor[(*shape, dtype)])
    chunks = x.split(sizes, axis=axis)
    assert flags(chunks) == expected


def test_a_blocked_axis_chunk_with_aligned_data_but_misaligned_scales_is_a_view(tr):
    mx = BlockScaled(e4m3, ue8m0, 32)
    first, second = tr.val(Tensor[1, 1024, mx]).split([32, 992], axis=1)
    # `second` starts 32 data bytes in (aligned) but at scale byte 1 (not aligned): counting scale
    # elements as data elements (a scale offset of 32 instead of cdiv(32, 32) = 1) would call it aligned
    assert first.type.contiguous
    assert not second.type.contiguous and second.type == Tensor[1, 992, mx].as_view()
    assert second.type.shape[1] == 992
