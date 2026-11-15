"""Static slices ``x[k]`` of a parameter in the row template (note 16 of the numerics contract): the Triton they lower to, the runtime
dimension scalars their offsets need, and the views that stay refused (rule ``view``). The bytes are checked on a GPU by
``test_gpu_slices.py``.
"""

import pytest
import slice_cases as sc
from test_lower import check_golden

import tilefish
from tilefish import Dim, InOut, Tensor, TilefishLoweringError, f32, i32
from tilefish.aot import compile_specs

R = Dim("R")
N = Dim("N")
M = Dim("M")
D = Dim("D", 128)


def rule_of(kernel, template=None):
    with pytest.raises(TilefishLoweringError) as e:
        tilefish.lower(kernel, template=template)
    err = e.value
    return {err.rule} | {c.rule for c in getattr(err, "causes", ())}


def test_golden_source():
    check_golden("vq_lookup", sc.vq_lookup.lower().source)


def test_a_row_of_a_plane_is_read_at_its_offset_with_a_runtime_dimension():
    low = sc.vq_lookup.lower()
    assert low.template == "row"
    assert "v0 = tl.load(codes + row)  # codes[0]" in low.source and "v8 = tl.load(codes + 3 * size_R + row)  # codes[3]" in low.source
    assert "# codes" not in low.source.replace("# codes[", "")  # the plane itself is never loaded
    (spec,) = compile_specs(low.extern)
    size = next(a for a in spec.args if a.name == "size_R")
    assert (size.kind, size.type, size.dim) == ("scalar", "i32", "R")  # a host derives it from the shapes, like any dimension scalar
    pre = [p for p in spec.preconditions if p.kind == "max_elements" and p.arg == "codes"]
    assert pre and pre[0].value == 2**31 - 1  # K * R within the i32 offsets, so k * R + row is too


def test_a_table_at_a_static_offset():
    low = sc.vq_tables.lower()
    assert "v3 = tl.load(tables + 128 * size_Nb + v3_i * D + cols, mask=mask, other=0)" in low.source
    assert [p.name for p in low.extern.signature.params][-5:] == ["D", "BLOCK_D", "tables_rows", "size_Nb", "size_R"]  # one rows parameter
    specs = compile_specs(low.extern)
    assert [s.stem for s in specs] == ["vq_tables.Nb64", "vq_tables.Nb200"]
    assert [next(a for a in s.args if a.name == "tables_rows").value for s in specs] == [64, 200]
    assert all(next(a for a in s.args if a.name == "size_Nb").dim == "Nb" for s in specs)


def test_per_row_per_column_and_full_slices():
    src = sc.plane_rows.lower().source
    assert "v1 = tl.load(w + size_R + row)  # w[1]" in src  # [R, 1] of a [3, R, 1] plane: one value per row
    assert "v3 = tl.load(bias + cols, mask=mask, other=0)  # bias[0]" in src  # [D] of a [2, D] parameter: a static offset
    assert "v6 = tl.load(bias + 128 + cols, mask=mask, other=0)  # bias[1]" in src
    assert "tl.store(out1 + row, v11)  # return value 1" in src
    assert "v1 = tl.load(xs + 128 * size_R + offs, mask=mask, other=0)  # xs[1]" in sc.slab_max.lower().source
    chunked = sc.pick_argmax.lower().source  # a slab of a long row is read chunk by chunk
    assert "        v1 = tl.load(logits + 3000 * size_R + offs, mask=mask, other=0)  # logits[1]" in chunked


def test_the_header_says_where_each_slice_is_read():
    assert "# 'codes[3]' [R] is read at codes + 3 * size_R (a static slice, note 16)" in sc.vq_lookup.lower().source


def test_a_slice_used_only_through_its_shape_is_not_read():
    @tilefish.kernel
    def rows_of(x: Tensor[R, D, f32], codes: Tensor[2, R, i32]) -> tuple[Tensor[R, D, f32], Tensor[R, i32]]:
        return x * 2.0, tilefish.iota(codes[1], axis=0)  # iota reads only the shape

    body = rows_of.lower().source.split("):\n", 1)[1]
    assert "codes" not in body and "tl.store(out1 + row, v2)  # return value 1" in body


def test_a_slice_is_lowered_by_the_row_template_only():
    assert rule_of(sc.vq_lookup, "elementwise") == {"elementwise.slice"}

    @tilefish.kernel
    def flat(codes: Tensor[2, N, i32]) -> Tensor[N, i32]:
        return codes[0] + codes[1]

    assert rule_of(flat) == {"no-template", "elementwise.slice", "row.no-rows"}


@pytest.mark.parametrize(
    "case, why",
    [
        ("column", "selects along axis 1"),
        ("computed", "is not a parameter of the kernel"),
        ("written", "the inout parameter 'codes' is written by the kernel"),
        ("element", "one element of the rank-1 parameter 'v'"),
        ("product", "not affine in the dimensions"),
    ],
)
def test_every_other_view_is_refused(case, why):
    if case == "column":

        @tilefish.kernel
        def k(x: Tensor[R, D, f32], codes: Tensor[R, 4, i32]) -> Tensor[R, i32]:
            return codes[:, 1] + 0

    elif case == "computed":

        @tilefish.kernel
        def k(x: Tensor[R, D, f32], codes: Tensor[2, R, i32]) -> Tensor[R, i32]:
            return (codes + 1)[0]

    elif case == "written":

        @tilefish.kernel
        def k(x: Tensor[R, D, f32], codes: InOut[Tensor[2, R, i32]]) -> Tensor[R, i32]:
            y = codes[0] + 1
            codes[...] = codes + 1
            return y

    elif case == "element":

        @tilefish.kernel
        def k(x: Tensor[R, D, f32], v: Tensor[8, f32]) -> Tensor[R, D, f32]:
            return x * v[3]

    else:

        @tilefish.kernel
        def k(x: Tensor[R, D, f32], planes: Tensor[2, R, M, f32]) -> Tensor[R, D, f32]:
            return x + tilefish.sum(planes[1], axis=-1, keepdims=True)

    with pytest.raises(TilefishLoweringError) as e:
        k.lower()
    assert e.value.rule == "view" and why in str(e.value) and "note 16" in str(e.value)


def test_a_returned_slice_is_refused_and_a_copy_of_it_is_not():
    @tilefish.kernel
    def returned(x: Tensor[R, D, f32], codes: Tensor[2, R, i32]) -> Tensor[R, i32]:
        return codes[0]  # at offset 0, so its type is a contiguous tensor

    @tilefish.kernel
    def copied(x: Tensor[R, D, f32], codes: Tensor[2, R, i32]) -> tuple[Tensor[R, D, f32], Tensor[R, i32]]:
        return x + 0.0, codes[1] + 0

    assert rule_of(returned) == {"view-value"}
    assert "v2 = tl.load(codes + size_R + row)  # codes[1]" in copied.lower().source


def test_a_slice_that_does_not_fit_the_rows():
    @tilefish.kernel
    def other(x: Tensor[R, D, f32], codes: Tensor[2, N, i32]) -> tuple[Tensor[R, D, f32], Tensor[N, i32]]:
        return x + 1.0, codes[1] + 0  # N values, a template of R rows

    assert "row.shape" in rule_of(other, "row")


def test_the_reference_mode_reads_the_slice():
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(0)
    codes = torch.randint(-2, 70, (4, 33), generator=gen, dtype=torch.int32)
    tables = [torch.randn(64, 128, generator=gen) for _ in range(4)]
    got = tilefish.run(sc.vq_lookup, codes, *tables, mode="reference")
    rows = [torch.where(((c >= 0) & (c < 64))[:, None], t[c.clamp(0, 63).long()], 0.0) for c, t in zip(codes, tables)]
    assert torch.equal(got, ((rows[0] + rows[1]) + rows[2]) + rows[3])
