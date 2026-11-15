"""A pointer parameter an extern does not use is declared ``Null[dtype]`` and passed as the null pointer at every launch.

A kernel that reads a pointer only when it is not null (a prebuilt kernel's ``if (bias)``) or only under a constexpr flag the declaration turns off is
declared with that parameter ``Null[f32]``: a caller does not pass it, no tensor is bound to it, it binds no dimension and has no precondition, and both
hosts write the 64-bit zero at its slot (manifest kind ``null``). Its element type is the pointer type of the spec and the manifest (``*fp32``). No GPU
here: the gpu tests launch it (``test_gpu_types.py``).
"""

import copy
import json

import pytest
from built_fakes import fake_build
from fake_torch import FakeTensor, prepare, verdict

import tilefish
from tilefish import Dim, Manifest, Null, Out, Tensor, TilefishBuildError, TilefishDeclarationError, TilefishManifestError, TilefishTypeError, const, f16, f32
from tilefish._binary import check_ptx_abi
from tilefish.aot import CompileSpec, compile_specs, declaration_digest, write_manifest
from tilefish.hosts import rust

n = Dim("n")


@tilefish.extern(triton="np_kernels:scale_bias", grid=lambda n: (tilefish.cdiv(n, 256),), name="scale_null")
def scale_null(x: Tensor[n, f32], bias: Null[f32], y: Out[Tensor[n, f32]], n_elem: n, HAS_BIAS: const = False, BLOCK: const = 256): ...


@tilefish.extern(triton="np_kernels:scale_bias", grid=lambda n: (tilefish.cdiv(n, 256),), name="scale_bias")
def scale_bias(x: Tensor[n, f32], bias: Tensor[n, f32], y: Out[Tensor[n, f32]], n_elem: n, HAS_BIAS: const = True, BLOCK: const = 256): ...


def test_the_annotation():
    assert repr(Null[f16]) == "Null[f16]"
    assert str(scale_null.signature.params[1]) == "bias: Null[f32]"
    assert scale_null.signature.params[1].kind == "null" and scale_null.signature.params[1].null_dtype == f32


@pytest.mark.parametrize("bad", [tilefish.bool_, "f32", 32, Tensor[n, f32]])
def test_the_dtype_is_a_storage_dtype(bad):
    with pytest.raises(TilefishDeclarationError, match=r"Null\[...\] takes the element dtype of the pointer"):
        Null[bad]


def test_only_an_extern_takes_one():
    with pytest.raises(TilefishDeclarationError, match="a pointer passed as null, which only an extern op takes"):

        @tilefish.kernel
        def k(x: Tensor[n, f32], b: Null[f32]) -> Tensor[n, f32]:
            return x

    with pytest.raises(TilefishDeclarationError, match="a pointer passed as null, which only an extern op takes"):

        @tilefish.graph
        def g(x: Tensor[n, f32], b: Null[f32]):
            return x

    with pytest.raises(TilefishDeclarationError, match="cannot have a default"):

        @tilefish.extern(triton="m:k", grid=(1,))
        def k2(x: Tensor[n, f32], b: Null[f32] = None): ...


def test_a_caller_does_not_pass_it():
    assert list(scale_null.signature.call_signature.parameters) == ["x", "HAS_BIAS", "BLOCK"]

    @tilefish.graph
    def g(x: Tensor[n, f32]):
        return scale_null(x)

    (node,) = g.graph.nodes
    assert node.callee is scale_null and len(node.inputs) == 1
    with pytest.raises(TilefishTypeError):

        @tilefish.graph
        def g2(x: Tensor[n, f32], b: Tensor[n, f32]):
            return scale_null(x, b)


def test_nothing_names_it():
    with pytest.raises(TilefishDeclarationError, match="not a tensor input of this op"):

        @tilefish.extern(triton="m:k", grid=(1,))
        def k(b: Null[f32], y: Out[Tensor[n, f32], tilefish.MayAlias("b")], n_elem: n): ...

    with pytest.raises(TilefishDeclarationError, match=r"is not a Strided\[...\] tensor parameter"):

        @tilefish.extern(triton="m:k", grid=(1,))
        def k2(x: Tensor[n, f32], b: Null[f32], s: tilefish.StrideOf("b", 0)): ...


def test_the_spec():
    (spec,) = compile_specs(scale_null)
    (arg,) = [a for a in spec.args if a.name == "bias"]
    assert (arg.kind, arg.type, arg.dtype) == ("null", "*fp32", "f32")
    assert spec.signature["bias"] == "*fp32" and (1,) not in spec.attrs  # a pointer argument, with no hint: nothing is loaded through it
    assert not any(p.arg == "bias" for p in spec.preconditions)
    assert [p.name for p in spec.launch_params] == ["x", "bias", "y", "n_elem", "global_scratch", "profile_scratch"]
    assert spec.schema == 3
    back = CompileSpec.from_json(spec.to_json())
    back.verify_against(scale_null)
    assert back.verified
    assert declaration_digest(scale_null) != declaration_digest(scale_bias)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda a: a.update(type="*fp16"), "a null pointer's type is the pointer type of its dtype"),
        (lambda a: a.update(dtype="f16"), "a null pointer's type is the pointer type of its dtype"),
        (lambda a: a.update(access="in"), "must have exactly the fields"),
        (lambda a: a.update(shape=["n"]), "must have exactly the fields"),
    ],
)
def test_a_spec_reader_holds_a_null_pointer_to_its_rules(change, message):
    (spec,) = compile_specs(scale_null)
    doc = spec.to_dict()
    change(next(a for a in doc["args"] if a["name"] == "bias"))
    with pytest.raises(tilefish.TilefishSpecError, match=message):
        CompileSpec.from_dict(doc)


def test_the_manifest_records_it(tmp_path):
    (spec,) = compile_specs(scale_null)
    write_manifest([fake_build(spec)], tmp_path / "manifest.json", cubins=True)
    entry = json.loads((tmp_path / "manifest.json").read_text())["kernels"][0]
    assert entry["abi"]["params"][1] == {"slot": 1, "index": 1, "name": "bias", "kind": "null", "type": "*fp32", "ptx_type": "u64", "size": 8, "dtype": "f32"}
    assert [p["slot"] for p in entry["abi"]["scratch"]] == [4, 5]
    Manifest.load(tmp_path).built_kernel(spec, "sm_90", pinned=True)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda p: p.update(type="*fp16"), "the type of a null pointer"),
        (lambda p: p.update(dtype="e9"), "the type of a null pointer"),
        (lambda p: p.update(ptx_type="u32", size=4), "a null pointer is passed as a u64 of 8 bytes"),
        (lambda p: p.update(access="in"), "it has a dtype and nothing a tensor or a scalar has"),
        (lambda p: p.update(shape=["n"]), "it has a dtype and nothing a tensor or a scalar has"),
        (lambda p: p.update(value=0), "it has a dtype and nothing a tensor or a scalar has"),
        (lambda p: p.update(dim="n"), "it has a dtype and nothing a tensor or a scalar has"),
    ],
)
def test_the_manifest_reader_holds_a_null_pointer_to_its_rules(tmp_path, change, message):
    (spec,) = compile_specs(scale_null)
    write_manifest([fake_build(spec)], tmp_path / "manifest.json", cubins=True)
    path = tmp_path / "manifest.json"
    doc = json.loads(path.read_text())
    change(doc["kernels"][0]["abi"]["params"][1])
    path.write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match=message):
        Manifest.load(tmp_path)


def test_a_launch_passes_zero_at_its_slot():
    x = FakeTensor((1000,), "f32", 1 << 44)
    prepared, error = prepare(scale_null, [x], next_slab=1 << 46)
    assert error is None
    assert "bias" not in prepared.tensors
    v = verdict(prepared, error)
    assert v["params"] == [1 << 44, 0, 1 << 46, 1000, 0, 0]  # x, the null bias, y, n_elem, the two scratch slots
    from tilefish import runtime

    assert [k for k, _, _ in runtime._launch_values(prepared)] == ["pointer", "pointer", "scalar", "constexpr", "constexpr"]  # the launch writes the zero


PTX = """.version 8.8
.target sm_90
.address_size 64

.visible .entry scale_bias_f32(
\t.param .u64 x,
\t.param .u64 bias,
\t.param .u64 out,
\t.param .u32 n
)
{
\tret;
}
"""


def test_a_prebuilt_kernel_takes_it_as_a_pointer(tmp_path):
    path = tmp_path / "k.ptx"
    path.write_text(PTX)

    @tilefish.extern(ptx=path, symbol="scale_bias_f32", block=(256,), grid=lambda n: (tilefish.cdiv(n, 256),))
    def k(x: Tensor[n, f32], bias: Null[f32], out: Out[Tensor[n, f32]], n_elem: n): ...

    (spec,) = compile_specs(k)
    check_ptx_abi(spec, PTX, "scale_bias_f32")
    assert [p.name for p in spec.launch_params] == ["x", "bias", "out", "n_elem"]
    prepared, error = prepare(k, [FakeTensor((10,), "f32", 1 << 44)], next_slab=1 << 46)
    assert verdict(prepared, error)["params"] == [1 << 44, 0, 1 << 46, 10]  # no scratch slots: a prebuilt binary's ABI is its parameters

    @tilefish.extern(ptx=path, symbol="scale_bias_f32", block=(256,), grid=lambda n: (tilefish.cdiv(n, 256),))
    def k_wrong(x: Tensor[n, f32], bias: Null[f32], out: Out[Tensor[n, f32]], n_elem: tilefish.i64[n]): ...

    with pytest.raises(TilefishBuildError, match="does not have the C ABI its stub declares"):
        check_ptx_abi(compile_specs(k_wrong)[0], PTX, "scale_bias_f32")


def test_a_plan_binds_no_buffer_to_it():
    @tilefish.graph
    def g(x: Tensor[n, f32], b: Tensor[n, f32]):
        return scale_bias(scale_null(x), b)

    p = tilefish.plan(g, buckets={"n": [256, 1000]})
    first, second = p.steps(n=1000)
    assert first["entry"] == "scale_null" and [a["name"] for a in first["args"]] == ["x", "y"]
    assert second["entry"] == "scale_bias" and [a["name"] for a in second["args"]] == ["x", "bias", "y"]


def test_the_rust_launcher_does_not_take_it(tmp_path):
    (spec,) = compile_specs(scale_null)
    write_manifest([fake_build(spec, arch="sm_103")], tmp_path / "manifest.json", cubins=True)
    m = Manifest.load(tmp_path)
    m.check_pin(m.digest)
    text = rust.render(m, ops=[scale_null])
    assert "Null pointers (every launch passes 0 for them" in text and "`bias` (*fp32)" in text
    launch = text[text.index("pub fn launch<'a>(") :]
    launch = launch[: launch.index(") -> Result")]
    assert "bias" not in launch and "x: TensorRef<'a, F32>" in launch and "y: TensorMut<'a, F32>" in launch


def test_the_rust_launcher_refuses_a_malformed_one(tmp_path):
    (spec,) = compile_specs(scale_null)
    write_manifest([fake_build(spec, arch="sm_103")], tmp_path / "manifest.json", cubins=True)
    doc = json.loads((tmp_path / "manifest.json").read_text())
    bad = copy.deepcopy(doc)
    bad["kernels"][0]["abi"]["params"][1]["access"] = "in"
    with pytest.raises(TilefishManifestError):
        rust._validate_entry(bad["kernels"][0])
    rust._validate_entry(doc["kernels"][0])
