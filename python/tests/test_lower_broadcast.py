"""The elementwise template broadcasting an operand over the leading axes (bytes: ``test_gpu_broadcast.py``)."""

import broadcast_cases as bc
import pytest
from test_lower import check_golden

import tilefish
from tilefish import Dim, InOut, Tensor, TilefishLoweringError, f32, i32
from tilefish.aot import compile_specs

M = Dim("M")
N = Dim("N")
D = Dim("D", 64)


def test_golden_source():
    check_golden("pos_add", bc.pos_add.lower().source)


def test_a_trailing_shape_is_read_at_the_flat_index_modulo_its_size():
    low = bc.pos_add.lower()
    assert low.template == "elementwise"
    assert "v1 = tl.load(pe + offs % pe_n, mask=mask, other=0)  # pe" in low.source
    (spec,) = compile_specs(low.extern)
    counts = {a.name: (a.dim, a.count_of) for a in spec.args if a.kind == "scalar"}
    assert counts["n"][1] == "y" and counts["pe_n"][1] == "pe"
    low2 = bc.scale_cols.lower()
    assert low2.template == "elementwise" and "w + offs % w_n" in low2.source and "b + offs % b_n" in low2.source


def test_the_row_template_keeps_the_kernels_it_takes():
    @tilefish.kernel
    def bias(x: Tensor[M, D, f32], b: Tensor[D, f32]) -> Tensor[M, D, f32]:
        return x + b

    assert bias.lower().template == "row"


def test_a_broadcast_operand_is_never_written_or_indexed():
    @tilefish.kernel
    def writes(y: Tensor[M, N, D, f32], acc: InOut[Tensor[N, D, f32]]) -> Tensor[M, N, D, f32]:
        acc[...] = acc + 1.0
        return y + acc

    @tilefish.kernel
    def returns(y: Tensor[M, N, D, f32], z: Tensor[N, D, f32]) -> tuple[Tensor[M, N, D, f32], Tensor[N, D, f32]]:
        return y + z, z + 1.0

    V = Dim("V")

    @tilefish.kernel
    def gathers(y: Tensor[M, N, D, f32], table: Tensor[V, f32], idx: Tensor[N, D, i32]) -> Tensor[M, N, D, f32]:
        return y + tilefish.gather_rows(table, idx, oob="zero")

    for k, rule in ((writes, "elementwise.broadcast-write"), (returns, "elementwise.broadcast-write"), (gathers, "elementwise.broadcast-index")):
        with pytest.raises(TilefishLoweringError) as e:
            k.lower(template="elementwise")
        assert e.value.rule == rule, k.name


def test_a_shape_that_does_not_broadcast_fits_no_template():
    @tilefish.kernel
    def two(y: Tensor[M, N, D, f32], z: Tensor[M, 3, f32]) -> tuple[Tensor[M, N, D, f32], Tensor[M, 3, f32]]:
        return y * 2.0, z * 2.0

    with pytest.raises(TilefishLoweringError) as e:
        two.lower()
    assert e.value.rule == "no-template"
    assert "parameter 'z' (Tensor[M, 3, f32]) does not broadcast to the shape [M, N, D]" in str(e.value)
