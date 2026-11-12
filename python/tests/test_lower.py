"""Golden sources of the lowering, and what the generated modules and their compile specs must look like.

Regenerate the snapshots in ``tests/golden/`` with ``UPDATE_GOLDEN=1 pytest tests/test_lower.py``
and review the diff: a changed snapshot is a change of the numerics contract (see ``tilefish.lower``).
"""

import ast
import os
import pathlib
import subprocess
import sys

import pytest
from conftest import REPO_ROOT, make_my_kernels, readme_snippet
from lower_cases import gelu_exact, layernorm, softmax, swiglu

import tilefish
from tilefish import LoweredKernel
from tilefish.aot import CompileSpec, compile_specs

GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "golden"
TESTS_DIR = pathlib.Path(__file__).resolve().parent
PTR16 = [["tt.divisibility", 16]]


@pytest.fixture(scope="module")
def readme_kernel():
    if not (REPO_ROOT / "README.md").exists():
        pytest.skip("README.md is not available (not running from a source checkout)")
    ns = {"my_kernels": make_my_kernels()}
    exec(compile(readme_snippet(), "README.md", "exec"), ns)
    return ns["residual_add_rmsnorm"]


@pytest.fixture(scope="module")
def cases(readme_kernel):
    """name -> kernel op, for every golden."""
    return {
        "residual_add_rmsnorm": readme_kernel,
        "layernorm": layernorm,
        "gelu_exact": gelu_exact,
        "swiglu": swiglu,
        "softmax": softmax,
    }


NAMES = ["residual_add_rmsnorm", "layernorm", "gelu_exact", "swiglu", "softmax"]


def check_golden(name: str, source: str) -> None:
    path = GOLDEN_DIR / f"{name}.py"
    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN_DIR.mkdir(exist_ok=True)
        path.write_bytes(source.encode("utf-8"))
        return
    assert path.exists(), f"no golden for {name}: run UPDATE_GOLDEN=1 pytest tests/test_lower.py and review {path}"
    expected = path.read_bytes().decode("utf-8")
    assert source == expected, f"{name} no longer matches {path}; if the change is intended, rerun with UPDATE_GOLDEN=1"


@pytest.mark.parametrize("name", NAMES)
def test_golden_source(cases, name):
    lowered = cases[name].lower()
    assert isinstance(lowered, LoweredKernel)
    check_golden(name, lowered.source)


@pytest.mark.parametrize("name", NAMES)
def test_lowering_is_deterministic(cases, name):
    first = cases[name].lower()
    again = tilefish.lower(cases[name])
    assert first.source == again.source
    assert first.extern.triton == again.extern.triton
    assert [s.to_json() for s in compile_specs(first.extern)] == [s.to_json() for s in compile_specs(again.extern)]


def test_lowering_does_not_depend_on_the_hash_seed():
    script = (
        "import sys, hashlib\n"
        f"sys.path.insert(0, {str(TESTS_DIR)!r})\n"
        "import lower_cases as c\n"
        "for k in (c.layernorm, c.gelu_exact, c.swiglu, c.softmax):\n"
        "    print(hashlib.sha256(k.lower().source.encode()).hexdigest())\n"
    )
    outs = []
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
        outs.append(done.stdout)
    assert outs[0] == outs[1] == outs[2]
    assert len(outs[0].split()) == 4


# -- what every generated module must look like -----------------------------------------------------------------------


def parse(lowered):
    return ast.parse(lowered.source, filename=lowered.filename)


@pytest.mark.parametrize("name", NAMES)
def test_source_compiles_and_imports_only_triton(cases, name):
    lowered = cases[name].lower()
    compile(lowered.source, f"{name}.py", "exec")  # syntax only: nothing is executed
    tree = parse(lowered)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module)
    assert sorted(imported) == ["triton", "triton.language"]
    # one module-level function and the two imports, nothing else
    kinds = [type(n).__name__ for n in tree.body]
    assert kinds == ["Import", "Import", "FunctionDef"]
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert fn.name == f"tf_{name}" == lowered.fn_name
    assert [ast.unparse(d) for d in fn.decorator_list] == ["triton.jit"]


@pytest.mark.parametrize("name", NAMES)
def test_source_uses_only_defined_names(cases, name):
    """Every name the kernel reads is a parameter, a local, ``tl`` or ``float``: Triton would fail on anything else."""
    lowered = cases[name].lower()
    (fn,) = [n for n in parse(lowered).body if isinstance(n, ast.FunctionDef)]
    defined = {a.arg for a in fn.args.args} | {"tl", "float"}
    nodes = [n for stmt in fn.body for n in ast.walk(stmt)]
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            assert node.id not in defined, f"{node.id} is assigned twice (SSA: one local per value)"
            defined.add(node.id)
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            assert node.id in defined, node.id
    # constexpr parameters are exactly the const parameters of the extern stub
    constexpr = {a.arg for a in fn.args.args if a.annotation is not None}
    assert all(ast.unparse(a.annotation) == "tl.constexpr" for a in fn.args.args if a.annotation is not None)
    stub = {p.name for p in lowered.extern.signature.consts}
    assert constexpr == stub
    assert [a.arg for a in fn.args.args] == [p.name for p in lowered.extern.signature.params]


@pytest.mark.parametrize("name", NAMES)
def test_casts_and_rounding_points_are_exactly_the_irs(cases, name):
    op = cases[name]
    lowered = op.lower()
    casts = [n for n in op.graph.nodes if n.op == "cast"]
    assert lowered.source.count(".to(") == len(casts)
    for n in casts:
        assert f".to(tl.{ {'bf16': 'bfloat16', 'f32': 'float32'}[n.attrs['to'].name] })" in lowered.source
    marked = [line for line in lowered.source.splitlines() if line.rstrip().endswith("# rounding point")]
    assert len(marked) == len(op.graph.rounding_points())
    assert all(".to(" in line for line in marked)
    # no other casts of any kind sneak in
    for forbidden in ("tl.cast", "astype", "float16(", "bfloat16("):
        assert forbidden not in lowered.source


@pytest.mark.parametrize("name", NAMES)
def test_every_loaded_tensor_is_loaded_in_its_storage_dtype(cases, name):
    """Loads are never pre-cast: each ``tl.load`` is assigned as is, and the only ``.to`` calls are the IR's casts."""
    lowered = cases[name].lower()
    for line in lowered.source.splitlines():
        if "tl.load(" in line:
            code = line.split("#")[0].strip()
            assert code.endswith(")") and ".to(" not in code
            assert "other=0" in code or " + row)" in code


@pytest.mark.parametrize("name", NAMES)
def test_write_puts_the_source_where_the_module_name_says(cases, name, tmp_path):
    lowered = cases[name].lower()
    path = lowered.write(tmp_path)
    assert path == tmp_path / "tilefish_generated" / f"{name}.py"
    assert path.read_bytes() == lowered.source.encode("utf-8")
    assert lowered.ref == f"tilefish_generated.{name}:tf_{name}"
    assert lowered.filename == f"tilefish_generated/{name}.py"


# -- compile_specs of the generated extern -----------------------------------------------------------------------------

EXPECTED = {
    "residual_add_rmsnorm": dict(
        template="row",
        signature={
            "x": "*bf16",
            "residual": "*fp32",
            "w": "*bf16",
            "eps": "constexpr",
            "out": "*bf16",
            "D": "constexpr",
            "BLOCK_D": "constexpr",
        },
        constants={(3,): 1e-6, (5,): 2560, (6,): 4096},
        attrs={(0,): PTR16, (1,): PTR16, (2,): PTR16, (4,): PTR16},
        grid="(M,)",
        num_warps=16,
    ),
    "layernorm": dict(
        template="row",
        signature={
            "x": "*bf16",
            "w": "*bf16",
            "b": "*bf16",
            "eps": "constexpr",
            "out": "*bf16",
            "D": "constexpr",
            "BLOCK_D": "constexpr",
        },
        constants={(3,): 1e-5, (5,): 768, (6,): 1024},
        attrs={(0,): PTR16, (1,): PTR16, (2,): PTR16, (4,): PTR16},
        grid="(M,)",
        num_warps=4,
    ),
    "gelu_exact": dict(
        template="elementwise",
        signature={"x": "*bf16", "out": "*bf16", "n": "i32", "BLOCK": "constexpr"},
        constants={(3,): 1024},
        attrs={(0,): PTR16, (1,): PTR16, (2,): PTR16},  # n = T is divisible by 16 because T is
        grid="(cdiv(T, 1024),)",
        num_warps=4,
    ),
    "swiglu": dict(
        template="elementwise",
        signature={"gate": "*bf16", "up": "*bf16", "out": "*bf16", "n": "i32", "BLOCK": "constexpr"},
        constants={(4,): 1024},
        attrs={(0,): PTR16, (1,): PTR16, (2,): PTR16, (3,): PTR16},  # n = 3072 * M
        grid="(3*M,)",  # 1024 divides 3072, so cdiv folds
        num_warps=4,
    ),
    "softmax": dict(
        template="row",
        signature={"x": "*fp32", "out": "*fp32", "D": "constexpr", "BLOCK_D": "constexpr"},
        constants={(2,): 1000, (3,): 1024},
        attrs={(0,): PTR16, (1,): PTR16},
        grid="(M,)",
        num_warps=4,
    ),
}


@pytest.mark.parametrize("name", NAMES)
def test_compile_specs_of_the_generated_extern(cases, name):
    lowered = cases[name].lower()
    want = EXPECTED[name]
    assert lowered.template == want["template"]
    (spec,) = compile_specs(lowered.extern)
    assert isinstance(spec, CompileSpec)
    assert spec.stem == spec.op == name
    assert spec.triton == f"tilefish_generated.{name}:tf_{name}"
    assert spec.signature == want["signature"]
    assert list(spec.signature) == list(want["signature"])  # the function's parameter order
    assert spec.constants == want["constants"]
    assert spec.attrs == want["attrs"]
    assert spec.grid == want["grid"] == lowered.grid
    assert spec.num_warps == want["num_warps"] == lowered.num_warps
    assert spec.block == (32 * want["num_warps"], 1, 1)
    # the typed stub and the source agree on the parameter list
    (fn,) = [n for n in parse(lowered).body if isinstance(n, ast.FunctionDef)]
    assert list(spec.signature) == [a.arg for a in fn.args.args]
    # the spec survives the strict JSON round trip
    assert CompileSpec.from_json(spec.to_json()) == CompileSpec.from_dict(spec.to_dict())
    assert CompileSpec.from_json(spec.to_json()).to_dict() == spec.to_dict()


@pytest.mark.parametrize("name", NAMES)
def test_specs_are_verified_and_usable_for_compiling(cases, name):
    lowered = cases[name].lower()
    specs = lowered.specs()
    assert [s.to_json() for s in specs] == [s.to_json() for s in compile_specs(lowered.extern)]
    for spec in specs:
        assert spec.verified and spec.declaration_digest
        inputs = spec.ast_source_inputs()  # refuses an unverified spec
        assert inputs.signature == EXPECTED[name]["signature"] and inputs.constants == EXPECTED[name]["constants"]
        # a generated kernel writes Triton's default contraction out, so that it is part of the spec
        assert inputs.options == {"num_warps": EXPECTED[name]["num_warps"], "num_stages": 3, "enable_fp_fusion": True}
        back = CompileSpec.from_json(spec.to_json())
        assert not back.verified
        with pytest.raises(tilefish.TilefishSpecError, match="verify_against"):
            back.ast_source_inputs()
        back.verify_against(lowered.extern)  # the generated stub is an ordinary declaration
        assert back.verified and back.ast_source_inputs().constants == inputs.constants


def test_a_typed_const_keeps_its_dtype_in_the_spec(cases):
    (spec,) = compile_specs(cases["layernorm"].lower().extern)
    eps = next(a for a in spec.args if a.name == "eps")
    assert eps.dtype == "f32" and eps.value == 1e-5 and isinstance(eps.value, float)
    d = next(a for a in spec.args if a.name == "D")
    assert d.dtype is None and d.value == 768 and isinstance(d.value, int)  # generated constexprs are untyped


def test_module_name_is_chosen_by_the_caller(cases):
    lowered = cases["softmax"].lower(module="my_pkg.kernels.softmax")
    assert lowered.ref == "my_pkg.kernels.softmax:tf_softmax"
    (spec,) = compile_specs(lowered.extern)
    assert (spec.triton_module, spec.triton_function) == ("my_pkg.kernels.softmax", "tf_softmax")
    assert lowered.source == cases["softmax"].lower().source  # the module name is not in the source


def test_row_specs_carry_static_sizes_and_the_elementwise_spec_carries_divisibility(cases):
    (row,) = compile_specs(cases["layernorm"].lower().extern)
    assert [(p.arg, p.axis, p.value) for p in row.preconditions if p.kind == "static_size"] == [
        ("x", 1, 768),
        ("w", 0, 768),
        ("b", 0, 768),
        ("out", 1, 768),
    ]
    (ew,) = compile_specs(cases["gelu_exact"].lower().extern)
    assert [(p.arg, p.dim, p.by) for p in ew.preconditions if p.kind == "divisible"] == [("n", "T", 16)]
    (sw,) = compile_specs(cases["swiglu"].lower().extern)
    # n = 3072 * M is a multiple of 16 whatever M is, and the spec says how the host computes it
    assert [(p.arg, p.dim, p.by) for p in sw.preconditions if p.kind == "divisible"] == [("n", "3072*M", 16)]
    assert next(a for a in sw.args if a.name == "n").dim == "3072*M"
    assert sw.attrs[(3,)] == PTR16
