import pathlib
import runpy

import pytest

import tilefish
from tilefish import Tensor, bf16

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def block():
    path = EXAMPLES / "transformer_block.py"
    if not path.exists():
        pytest.skip("examples are not available")
    return runpy.run_path(str(path))


def test_transformer_block_traces_and_type_checks(block):
    g = block["transformer_block"].graph
    T, D, F = block["T"], block["D"], block["F"]
    assert g.ops() == [
        "rmsnorm",
        "gemm",
        "split",
        "attention",
        "gemm",
        "add_rmsnorm",
        "gemm",
        "swiglu",
        "gemm_mxfp8",
        "add",
        "store",
    ]
    assert g.inputs[0].access == "inout"
    assert g.outputs == (g.inputs[0],)
    assert g.outputs[0].type == Tensor[T, D, bf16]
    assert g.nodes[2].output_types == (Tensor[T, D, bf16],) * 3  # q, k, v
    assert g.nodes[7].output_types == (Tensor[T, F, bf16],)
    assert g.requirements == ()
    assert g.symbols == (T,)
    assert g.rounding_points() == ()


def test_gelu_variant_traces_too(block):
    g = block["transformer_block_gelu"].graph
    assert "gelu" in g.ops() and "swiglu" not in g.ops()


def test_extern_specs_of_the_example(block):
    for op in block["EXTERNS"]:
        specs = tilefish.aot.compile_specs(op)
        assert len(specs) == 1
        for spec in specs:
            assert spec.triton == f"my_kernels:{op.name}"
            assert spec.launch_params[-2].name == "global_scratch"
    (gelu,) = tilefish.aot.compile_specs(block["gelu"])
    assert gelu.grid == "(cdiv(5504*T, 1024),)"
    assert gelu.attrs[(2,)] == [["tt.divisibility", 16]]  # 5504 * T is a multiple of 16
    (attention,) = tilefish.aot.compile_specs(block["attention"])
    assert attention.signature["n_tokens"] == "i32"
    assert attention.constants == {(5,): 128, (6,): 0.08838834764831845}
    assert (4,) not in attention.attrs  # T has no divisibility, so n_tokens carries no hint
