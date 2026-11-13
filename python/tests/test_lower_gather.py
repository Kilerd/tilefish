"""``gather_rows`` and ``scatter_rows``: what they trace to, what they refuse (by rule), and the Triton they lower to.

The bytes are checked on a GPU by ``test_gpu_gather.py``.
"""

import gather_cases as gc
import pytest
from test_lower import check_golden

import tilefish
from tilefish import DevScalar, Dim, InOut, Out, Tensor, TilefishLoweringError, TilefishTypeError, bf16, f32, i32, i64
from tilefish.aot import compile_specs

V = Dim("V")
R = Dim("R")
D = Dim("D", 64)


# -- tracing ------------------------------------------------------------------------------------------------------------------


def test_gather_rows_has_the_index_shape_and_the_rows_of_the_table(tr):
    table = tr.val(Tensor[V, D, bf16])
    ids = tr.val(Tensor[R, i64])
    out = tilefish.gather_rows(table, ids, oob="zero")
    assert out.type == Tensor[R, D, bf16]
    (node,) = tr.ctx.nodes
    assert node.op == "gather_rows" and node.inputs == (table, ids) and node.attrs == {"oob": "zero"}
    flat = tr.val(Tensor[V, f32])
    grid = tr.val(Tensor[R, D, i32])
    assert tilefish.gather_rows(flat, grid, oob="clamp").type == Tensor[R, D, f32]


def test_a_gather_says_what_an_index_out_of_range_does(tr):
    table = tr.val(Tensor[V, D, bf16])
    ids = tr.val(Tensor[R, i32])
    for bad, match in (
        (lambda: tilefish.gather_rows(table, ids), "say what an index out of range does: oob='zero' or 'clamp'"),
        (lambda: tilefish.gather_rows(table, ids, oob="wrap"), "oob must be 'zero' or 'clamp'"),
        (lambda: tilefish.gather_rows(table, ids, oob="skip"), "oob must be 'zero' or 'clamp'"),
        (lambda: tilefish.gather_rows(table, tr.val(Tensor[R, f32]), oob="zero"), "an index is an integer tensor"),
        (lambda: tilefish.gather_rows(table, tilefish.load(tr.val(tilefish.DevScalar[i32])), oob="zero"), "has rank 0"),
        (lambda: tilefish.gather_rows(table, ids > 0, oob="zero"), "an index is an integer tensor"),
        (lambda: tilefish.gather_rows(ids > 0, ids, oob="zero"), "is a mask"),
    ):
        with pytest.raises(TilefishTypeError, match=match):
            bad()


def test_scatter_rows_writes_a_new_version_of_an_inout_parameter(tr):
    dst = tr.val(Tensor[V, D, bf16], name="dst", access="inout")
    idx = tr.val(Tensor[R, i64])
    src = tr.val(Tensor[R, D, bf16])
    assert tilefish.scatter_rows(dst, idx, src, oob="skip") is None
    (node,) = tr.ctx.nodes
    assert node.op == "scatter_rows" and node.attrs == {"target": "dst", "oob": "skip"} and node.inputs == (dst, idx, src)
    assert dst.live is node.outputs[0] and dst in tr.ctx.written


def test_a_scatter_refuses_what_it_cannot_write(tr):
    dst = tr.val(Tensor[V, D, bf16], name="dst", access="inout")
    idx = tr.val(Tensor[R, i64])
    src = tr.val(Tensor[R, D, bf16])
    for bad, match in (
        (lambda: tilefish.scatter_rows(dst, idx, src), "oob='skip'"),
        (lambda: tilefish.scatter_rows(dst, idx, src, oob="zero"), "oob must be 'skip'"),
        (lambda: tilefish.scatter_rows(tr.val(Tensor[V, D, bf16], access="out"), idx, src, oob="skip"), "an Out parameter"),
        (lambda: tilefish.scatter_rows(tr.val(Tensor[V, D, bf16]), idx, src, oob="skip"), "not an InOut parameter"),
        (lambda: tilefish.scatter_rows(dst, idx, tr.val(Tensor[R, D, f32]), oob="skip"), "the source is"),
        (lambda: tilefish.scatter_rows(dst, idx, tr.val(Tensor[V, D, bf16]), oob="skip"), "the source is"),
    ):
        with pytest.raises(TilefishTypeError, match=match):
            bad()


# -- lowering -----------------------------------------------------------------------------------------------------------------


GOLDEN = {"embed_masked": gc.embed_masked, "index_copy": gc.index_copy, "take_elems": gc.take_elems}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_source(name):
    check_golden(name, GOLDEN[name].lower().source)


def test_the_rows_of_a_table_are_a_runtime_scalar_a_constant_or_a_constant_per_bucket():
    (spec,) = compile_specs(gc.embed_lookup.lower().extern)
    rows = next(a for a in spec.args if a.name == "table_rows")
    assert (rows.kind, rows.type, rows.dim) == ("scalar", "i32", "V")
    (spec,) = compile_specs(gc.take_static.lower().extern)
    assert next(a for a in spec.args if a.name == "table_rows").value == 1000
    specs = compile_specs(gc.take_clamp.lower().extern)
    assert [s.stem for s in specs] == ["take_clamp.Vb64", "take_clamp.Vb200"]
    assert [next(a for a in s.args if a.name == "table_rows").value for s in specs] == [64, 200]


def test_a_table_is_not_loaded_by_the_prologue_and_has_a_max_elements_precondition():
    low = gc.embed_lookup.lower()
    assert "# table" not in low.source  # only the gather reads it
    (spec,) = low.specs()
    pre = [p for p in spec.preconditions if p.kind == "max_elements" and p.arg == "table"]
    assert pre and pre[0].value == 2**31 - 1  # N * D within the i32 offsets


def test_zero_masks_the_load_and_clamp_clamps_the_index():
    zero = gc.embed_lookup.lower().source
    assert "v1_ok = (v0 >= 0) & (v0 < table_rows)" in zero
    assert "tl.load(table + v1_i * D + cols, mask=mask & v1_ok, other=0)" in zero
    clamp = gc.take_clamp.lower().source
    assert "v1_i = tl.minimum(tl.maximum(v0, 0), table_rows - 1)" in clamp and "mask=mask, other=0)" in clamp


def test_a_per_row_gather_from_a_flat_table():
    src = gc.scale_rows.lower().source
    assert "v2 = tl.load(scales + v1, mask=v2_ok, other=0)" in src and "v3 = v0 * v2" in src


def test_a_scatter_masks_the_store_and_writes_nothing_else():
    src = gc.index_copy.lower().source
    assert "tl.store(dst + s_dst_i * D + cols, v1, mask=mask & s_dst_ok)" in src
    assert src.count("tl.store(") == 1 and "# dst" not in src
    elems = gc.scatter_elems.lower().source
    assert "tl.store(dst + v0, v1, mask=mask & s_dst_ok)" in elems


def test_the_elementwise_count_is_never_a_tables():
    (spec,) = compile_specs(gc.take_elems.lower().extern)
    assert next(a for a in spec.args if a.name == "n").count_of == "idx"


def test_a_table_that_is_also_an_input_is_loaded_and_gathered():
    src = gc.permute_rows.lower().source
    assert "v0 = tl.load(x + offs, mask=mask, other=0)  # x" in src and "tl.load(x + v2_i * D + cols" in src


def rule_of(kernel, template=None):
    with pytest.raises(TilefishLoweringError) as e:
        tilefish.lower(kernel, template=template)
    err = e.value
    return {err.rule} | {c.rule for c in getattr(err, "causes", ())}


def test_a_gather_reads_a_read_only_parameter():
    @tilefish.kernel
    def computed(x: Tensor[V, D, f32], ids: Tensor[R, i32]) -> Tensor[R, D, f32]:
        return tilefish.gather_rows(x * 2.0, ids, oob="zero")

    @tilefish.kernel
    def written(x: InOut[Tensor[V, D, f32]], ids: Tensor[R, i32]) -> Tensor[R, D, f32]:
        y = tilefish.gather_rows(x, ids, oob="zero")
        x[...] = x + 1.0
        return y

    assert rule_of(computed) == {"gather.table"}
    assert rule_of(written) == {"gather.table"}


def test_a_scatter_is_the_only_use_of_its_target():
    @tilefish.kernel
    def read_too(dst: InOut[Tensor[R, D, f32]], idx: Tensor[R, i32], src: Tensor[R, D, f32]) -> Tensor[R, D, f32]:
        y = dst + 0.0
        tilefish.scatter_rows(dst, idx, src, oob="skip")
        return y

    @tilefish.kernel
    def source_is_table(dst: InOut[Tensor[V, D, f32]], idx: Tensor[R, i32], src: Tensor[R, D, f32]) -> Tensor[R, D, f32]:
        y = tilefish.gather_rows(src, idx, oob="zero")  # src is a table and the source of the scatter: both read it
        tilefish.scatter_rows(dst, idx, src, oob="skip")
        return y

    @tilefish.kernel
    def stored_then(dst: InOut[Tensor[R, D, f32]], idx: Tensor[R, i32], src: Tensor[R, D, f32]):
        dst[...] = src
        tilefish.scatter_rows(dst, idx, src, oob="skip")

    @tilefish.kernel
    def twice(dst: InOut[Tensor[V, D, f32]], idx: Tensor[R, i32], src: Tensor[R, D, f32]):
        tilefish.scatter_rows(dst, idx, src, oob="skip")
        tilefish.scatter_rows(dst, idx, src, oob="skip")

    @tilefish.kernel
    def returned(dst: InOut[Tensor[V, D, f32]], idx: Tensor[R, i32], src: Tensor[R, D, f32]) -> Tensor[V, D, f32]:
        tilefish.scatter_rows(dst, idx, src, oob="skip")
        return dst

    assert gc.index_copy.lower().template == "row"
    assert rule_of(read_too) == {"scatter.target"}
    assert source_is_table.lower().template == "row"
    assert rule_of(stored_then) == {"scatter.target"}
    assert rule_of(twice) == {"scatter.target"}
    assert rule_of(returned) == {"scatter.target"}


def test_the_row_template_needs_rows_of_its_width_and_one_row_per_index():
    @tilefish.kernel
    def narrow(table: Tensor[V, 32, bf16], ids: Tensor[R, i32], x: Tensor[R, D, bf16]) -> tuple[Tensor[R, 32, bf16], Tensor[R, D, bf16]]:
        return tilefish.gather_rows(table, ids, oob="zero"), x + x

    @tilefish.kernel
    def same_cols(dst: InOut[Tensor[V, f32]], idx: Tensor[D, i32], src: Tensor[D, f32], x: Tensor[R, D, f32]) -> Tensor[R, D, f32]:
        tilefish.scatter_rows(dst, idx, src, oob="skip")
        return x + 1.0

    @tilefish.kernel
    def cube(table: Tensor[V, D, 2, bf16], ids: Tensor[R, i32]) -> Tensor[R, D, 2, bf16]:
        return tilefish.gather_rows(table, ids, oob="zero")

    assert rule_of(narrow, "row") == {"row.shape"}  # a gathered row is a row of the template
    assert rule_of(same_cols, "row") == {"row.scatter-index"}
    assert rule_of(cube) == {"table-rank"}


def test_the_elementwise_template_gathers_elements_only():
    assert rule_of(gc.embed_lookup, "elementwise") == {"elementwise.table-rank"}
    assert gc.take_elems_i64.lower().template == "elementwise"


def test_gather_and_scatter_in_one_kernel():
    @tilefish.kernel
    def move(dst: InOut[Tensor[V, D, bf16]], table: Tensor[V, D, bf16], src_ids: Tensor[R, i64], dst_ids: Tensor[R, i64], n: DevScalar[i32]):
        rows = tilefish.gather_rows(table, src_ids, oob="zero")
        keep = tilefish.iota(rows, axis=0) < tilefish.load(n)
        tilefish.scatter_rows(dst, dst_ids, tilefish.where(keep, rows, 0.0), oob="skip")

    low = move.lower()
    assert low.template == "row" and "dst_rows" in low.source and "table_rows" in low.source
    assert [p.name for p in low.extern.signature.params][-4:] == ["D", "BLOCK_D", "dst_rows", "table_rows"]


def test_an_out_parameter_is_not_a_scatter_target():
    with pytest.raises(TilefishTypeError, match="an Out parameter"):

        @tilefish.kernel
        def k(dst: Out[Tensor[R, D, f32]], idx: Tensor[R, i32], src: Tensor[R, D, f32]):
            tilefish.scatter_rows(dst, idx, src, oob="skip")

