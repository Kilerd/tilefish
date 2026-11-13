"""The code under "What it looks like" in the top-level README must run against the implementation."""

import pytest
from conftest import REPO_ROOT, make_my_kernels, readme_snippet

import tilefish
from tilefish import Dim, Tensor, bf16, f32


@pytest.fixture(scope="module")
def ns():
    if not (REPO_ROOT / "README.md").exists():
        pytest.skip("README.md is not available (not running from a source checkout)")
    namespace = {"my_kernels": make_my_kernels()}
    exec(compile(readme_snippet(), "README.md", "exec"), namespace)
    return namespace


def test_snippet_runs_and_defines_everything(ns):
    for name in ("residual_add_rmsnorm", "gelu", "gemm", "mlp"):
        assert name in ns, name
    assert isinstance(ns["residual_add_rmsnorm"], tilefish.KernelOp)
    assert isinstance(ns["gelu"], tilefish.ExternOp)
    assert isinstance(ns["gemm"], tilefish.LibraryOp)
    assert isinstance(ns["mlp"], tilefish.GraphOp)


def test_readme_mlp(ns):
    g = ns["mlp"].graph
    M, H, F = ns["M"], ns["H"], ns["F"]
    assert g.ops() == ["gemm", "split", "cast", "silu", "cast", "mul", "cast", "gemm"]
    assert g.outputs[0].type == Tensor[M, H, bf16]
    assert g.nodes[1].output_types == (Tensor[M, F, bf16].as_view(),) * 2
    assert len(g.rounding_points()) == 1


def test_readme_kernel(ns):
    g = ns["residual_add_rmsnorm"].graph
    assert len(g.rounding_points()) == 1
    assert g.rounding_points()[0] is g.nodes[-1]


def test_readme_gelu_extern_produces_a_spec(ns):
    (spec,) = tilefish.aot.compile_specs(ns["gelu"])
    assert spec.triton == "my_kernels:gelu"
    assert spec.signature == {"x": "*bf16", "y": "*bf16", "n_elem": "i32", "BLOCK": "constexpr"}
    assert spec.constants == {(3,): 1024}
    assert spec.attrs == {
        (0,): [["tt.divisibility", 16]],
        (1,): [["tt.divisibility", 16]],
        (2,): [["tt.divisibility", 16]],
    }
    assert spec.grid == "(cdiv(n, 1024),)"
    assert spec.num_warps == 4


def test_readme_dims_and_formats(ns):
    assert ns["D"] == 2560 and ns["F"] == 3072
    assert ns["K"].divisible_by == 32 and ns["N"].divisible_by == 128
    assert ns["mxfp8"].block == 32 and ns["nvfp4"].global_scale is f32
    assert ns["mxfp4"] != ns["mxfp8"]
    assert isinstance(ns["M"], Dim) and ns["M"].is_dynamic


def test_the_masks_section_runs_and_lowers():
    import re

    if not (REPO_ROOT / "README.md").exists():
        pytest.skip("README.md is not available (not running from a source checkout)")
    m = re.search(r"## Masks, indices, gathers and copies\n.*?```python\n(.*?)```", (REPO_ROOT / "README.md").read_text(), re.S)
    assert m, "README.md has no python block under '## Masks, indices, gathers and copies'"
    namespace: dict = {}
    exec(compile(m.group(1), "README.md", "exec"), namespace)
    assert namespace["embed"].lower().template == "row" and "static_range" in namespace["greedy"].lower().source
