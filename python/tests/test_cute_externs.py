"""CuTe DSL kernels as externs (``@tilefish.extern(cute=...)``, :mod:`tilefish._cute`, ``docs/cute.md``), without a GPU and without the DSL.

The DSL's version is the one thing a declaration reads of it (``importlib.metadata``): a fixture stands in for it. The build (the DSL's compile and
the captured launches) is replaced by what it hands on, the captured parameters and launch facts (:func:`tilefish._cute.classify_params`,
:func:`tilefish._cute.launch_of`) and a ``BuiltKernel`` made from them; ``test_gpu_cute.py`` runs the real thing.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import ctypes
import hashlib
import json
import pathlib
import struct
from dataclasses import dataclass

import pytest

from elf_builder import arch_header, cubin_like

import tilefish
from tilefish import Dim, Out, Strided, Tensor, const, f16, f32
from tilefish import _cuda, _cute
from tilefish._build import BuiltKernel, LaunchAttributes, write_manifest
from tilefish._elf import code_digest
from tilefish._manifest import Manifest, manifest_digest
from tilefish.aot import CompileSpec, compile_specs, declaration_digest
from tilefish.errors import TilefishBuildError, TilefishDeclarationError, TilefishManifestError, TilefishRuntimeError

DSL = "4.8.1"


@pytest.fixture(autouse=True)
def dsl(monkeypatch):
    monkeypatch.setattr(_cute, "dsl_version", lambda: DSL)


def host(a, b, c, stream):
    """Stands in for a ``@cute.jit`` host function (its module file is what the declaration digests)."""


def other_host(a, b, c, stream):
    """Another host function, of the same file."""


class Gemm:
    """Stands in for a kernel class of the CUTLASS examples, whose ``__call__`` is the host function."""

    def __call__(self, a, b, c, stream):
        pass


M = Dim("M", buckets=[128, 256])


def declare(cute=host, name="gemm", **kw):
    @tilefish.extern(cute=cute, name=name, **kw)
    def op(a: Tensor[M, 64, f16], b: Tensor[64, 64, f16], c: Out[Tensor[M, 64, f16]]): ...

    return op


# -- the declaration ---------------------------------------------------------------------------------------------------------------------------


def test_a_cute_extern_has_one_spec_per_bucket_of_its_static_shapes():
    op = declare()
    specs = compile_specs(op)
    assert [s.stem for s in specs] == ["gemm.M128", "gemm.M256"]
    s = specs[0]
    assert s.binary == {
        "kind": "cute",
        "host": f"{__name__}:host",
        "source_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "cutlass_dsl": DSL,
        "options": "",
    }
    assert s.triton == "" and s.num_warps == 0 and s.attrs == {} and s.constants == {}
    assert s.grid == "(1,)" and s.block == (1, 1, 1)  # placeholders: the compiled kernel's are the entry's
    assert [(a.name, a.kind, a.access, a.dtype, a.shape) for a in s.args] == [
        ("a", "pointer", "in", "f16", ("128", "64")),
        ("b", "pointer", "in", "f16", ("64", "64")),
        ("c", "pointer", "out", "f16", ("128", "64")),
    ]
    kinds = sorted({p.kind for p in s.preconditions})
    assert kinds == ["bucket", "max_elements", "pointer_alignment", "static_size"]
    assert all(p.bytes == 16 for p in s.preconditions if p.kind == "pointer_alignment")
    again = CompileSpec.from_json(s.to_json())  # a spec of a CuTe DSL kernel round-trips and verifies against its declaration
    again.verify_against(op)
    assert again.verified


def test_the_declaration_digest_has_the_dsl_the_host_file_and_the_options(monkeypatch):
    base = declaration_digest(declare())
    assert declaration_digest(declare()) == base
    assert declaration_digest(declare(cute=other_host)) != base  # another host function
    assert declaration_digest(declare(cute_options="--opt-level 2")) != base
    monkeypatch.setattr(_cute, "dsl_version", lambda: "4.8.0")
    assert declaration_digest(declare()) != base  # another DSL is another declaration


def test_an_object_whose_call_is_the_host_function_is_named_by_its_class():
    op = declare(cute=Gemm())
    assert op.binary.ref == f"{__name__}:Gemm()"
    assert compile_specs(op)[0].binary["host"] == f"{__name__}:Gemm()"
    assert op.node_attrs() == {"cute": f"{__name__}:Gemm()"}


def test_cute_args_and_options_are_kept_and_the_options_normalised():
    op = declare(cute_args=lambda a, b, c: (a, b, c), cute_options="  --opt-level   3 ")
    assert op.binary.options == "--opt-level 3"
    assert op.binary.args is not None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(grid=(1,)), "grid= is not an argument of a CuTe DSL kernel"),
        (dict(num_warps=4), "num_warps= is not an argument of a CuTe DSL kernel"),
        (dict(block=(128,)), "block= is not an argument"),
        (dict(cluster=(2, 1, 1)), "cluster= is not an argument"),
        (dict(shared=1024), "shared= is not an argument"),
        (dict(persistent=tilefish.persistent()), "persistent= is not an argument"),
        (dict(cute_options="--keep-cubin"), "may not give --keep-cubin"),
        (dict(cute_options="--gpu-arch=sm_90a"), "may not give --gpu-arch"),
        (dict(cute_options="--generate-line-info"), "may not give --generate-line-info"),
        (dict(cute_options="a;b"), "cute_options= is a string of CuTe DSL compile options"),
        (dict(cute_args="no"), "cute_args= makes the host function's arguments"),
    ],
)
def test_a_cute_declaration_is_checked(kwargs, message):
    with pytest.raises(TilefishDeclarationError, match=message):
        declare(**kwargs)


def test_cute_args_without_cute_is_refused():
    with pytest.raises(TilefishDeclarationError, match="cute_args= and cute_options= are for a CuTe DSL kernel"):

        @tilefish.extern(triton="m:f", cute_args=lambda a: (a,), grid=(1,))
        def op(a: Tensor[16, f32]): ...


def test_a_host_that_is_not_callable_is_refused():
    with pytest.raises(TilefishDeclarationError, match="cute= is a @cute.jit host function"):
        declare(cute="module:host")


def test_the_dsl_must_be_installed(monkeypatch):
    monkeypatch.setattr(_cute, "dsl_version", lambda: None)
    with pytest.raises(TilefishDeclarationError, match="needs the CuTe DSL \\(nvidia-cutlass-dsl\\) installed"):
        declare()


N = Dim("N")


@pytest.mark.parametrize(
    "stub, message",
    [
        (lambda: (lambda a: None, {"a": Tensor[N, 64, f16]}), "dimension\\(s\\) \\['N'\\] that have no buckets"),
        (lambda: (lambda a, n: None, {"a": Tensor[M, 64, f16], "n": M}), "parameter 'n' is not a tensor"),
        (lambda: (lambda a, k=1.0: None, {"a": Tensor[64, f16], "k": const[f32]}), "parameter 'k' is not a tensor"),
        (lambda: (lambda a: None, {"a": Strided[Tensor[64, f16]]}), "parameter 'a' is a strided view"),
    ],
)
def test_the_stub_of_a_cute_kernel_is_its_tensors_of_static_shapes(stub, message):
    fn, annotations = stub()
    fn.__annotations__ = annotations
    fn.__name__ = "op"
    with pytest.raises(TilefishDeclarationError, match=message):
        tilefish.extern(cute=host)(fn)


def test_an_expression_of_bucketed_dimensions_is_a_static_shape():
    @tilefish.extern(cute=host)
    def twice(a: Tensor[2 * M, 64, f16], b: Out[Tensor[M, 64, f16]]): ...

    assert [s.args[0].shape for s in compile_specs(twice)] == [("256", "64"), ("512", "64")]


def test_an_implementation_per_architecture_may_be_a_cute_kernel(tmp_path):
    @tilefish.extern(by_arch={"sm_103": tilefish.impl(cute=host), "sm_120": tilefish.impl(cute=other_host, cute_options="--opt-level 2")})
    def gemm(a: Tensor[M, 64, f16], b: Tensor[64, 64, f16], c: Out[Tensor[M, 64, f16]]): ...

    s103, s120 = compile_specs(gemm, arch=103), compile_specs(gemm, arch=120)
    assert s103[0].binary["host"].endswith(":host") and s103[0].implementation == "sm_103"
    assert s120[0].binary["options"] == "--opt-level 2" and s120[0].implementation == "sm_120"
    assert s103[0].declaration_digest != s120[0].declaration_digest
    assert repr(gemm.implementations["sm_103"]).startswith("<extern") or gemm.implementations["sm_103"].binary.kind == "cute"


def test_an_implementation_of_a_cute_kernel_takes_no_launch_arguments():
    with pytest.raises(TilefishDeclarationError, match="grid= is not an argument of a CuTe DSL kernel"):
        tilefish.impl(cute=host, grid=(1,))
    with pytest.raises(TilefishDeclarationError, match="cute_args= and cute_options= are for a CuTe DSL kernel"):
        tilefish.impl(triton="m:f", cute_options="--opt-level 2")


# -- the launch, from captured parameters ------------------------------------------------------------------------------------------------------


@dataclass
class Arg:
    """What classify_params reads of a tensor argument of a spec (an ArgSpec's fields)."""

    index: int
    name: str
    dtype: str = "f16"
    access: str = "in"
    shape_dims: tuple = (128, 64)


TENSORS = [Arg(0, "a"), Arg(1, "b", shape_dims=(64, 64)), Arg(2, "c", access="out")]
ADDRESSES = [
    {"a": 0x7F0000100000, "b": 0x7F0000200000, "c": 0x7F0000300000},
    {"a": 0x7F0001100000, "b": 0x7F0001200000, "c": 0x7F0001300000},
    {"a": 0x7F0002100000, "b": 0x7F0002200000, "c": 0x7F0002300000},
]
WORDS = bytes(range(8, 64))  # the bytes 8 to 63 of a descriptor the DSL encoded: constant for a static shape


def descriptor(address, garbage):
    """The 128 bytes the DSL's host code passes for a TMA descriptor: the address, its constant fields, and whatever was on its stack after them."""
    return struct.pack("<Q", address) + WORDS + bytes([garbage]) * 64


def captures(layout_of):
    """Three captures of the parameters ``layout_of(t, addresses)`` gives."""
    return [layout_of(t, addr) for t, addr in enumerate(ADDRESSES)]


GEMM_LAYOUT = [(64, 32), (128, 64), (128, 64), (128, 64)]


def gemm_params(t, addr):
    return [bytes(64), descriptor(addr["a"], t + 1), descriptor(addr["b"], 7 * t), descriptor(addr["c"], 255 - t)]


def test_the_parameters_of_a_tma_gemm_are_constant_bytes_and_three_descriptors():
    params = _cute.classify_params("gemm.M128", TENSORS, GEMM_LAYOUT, captures(gemm_params), ADDRESSES)
    assert [(p["slot"], p["kind"], p["name"]) for p in params] == [(0, "bytes", "param_0"), (1, "tensormap", "a"), (2, "tensormap", "b"), (3, "tensormap", "c")]
    assert params[0] == {"slot": 0, "index": 0, "name": "param_0", "kind": "bytes", "type": "bytes", "ptx_type": "b8[64]", "size": 64, "value": "00" * 64}
    a = params[1]
    assert a["template"] == "00" * 8 + WORDS.hex() + "00" * 64  # the address and the bytes the DSL does not write: zero
    assert (a["index"], a["dtype"], a["access"], a["shape"], a["size"], a["ptx_type"]) == (0, "f16", "in", [128, 64], 128, "b8[128]")
    assert params[3]["access"] == "out"
    assert _cute.descriptor(a["template"], 0x7F0000400000)[:64] == struct.pack("<Q", 0x7F0000400000) + WORDS


def test_the_parameters_of_a_plain_kernel_are_pointers():
    layout = [(8, 8), (8, 8), (8, 8), (4, 4)]
    params = _cute.classify_params(
        "add", TENSORS, layout, captures(lambda t, addr: [struct.pack("<Q", addr[n]) for n in "abc"] + [struct.pack("<i", 1024)]), ADDRESSES
    )
    assert [(p["kind"], p["name"], p.get("type")) for p in params] == [
        ("pointer", "a", "*fp16"), ("pointer", "b", "*fp16"), ("pointer", "c", "*fp16"), ("bytes", "param_3", "bytes"),
    ]  # fmt: skip
    assert params[3]["value"] == struct.pack("<i", 1024).hex()


@pytest.mark.parametrize(
    "layout, make, message",
    [
        # a byte that differs between the captures and is no tensor's address (a value of the host function's own)
        ([(8, 8)] * 3 + [(8, 8)], lambda t, addr: [struct.pack("<Q", addr[n]) for n in "abc"] + [struct.pack("<Q", 1000 + t)], "differ between the captured launches"),
        # garbage after byte 64 of a parameter that is not a descriptor (no address first)
        ([(128, 64)] + [(8, 8)] * 3, lambda t, addr: [bytes(64) + bytes([t]) * 64] + [struct.pack("<Q", addr[n]) for n in "abc"], "differ between the captured launches"),
        # an address inside a structure
        ([(16, 8), (8, 8), (8, 8)], lambda t, addr: [struct.pack("<QQ", 5, addr["a"]), struct.pack("<Q", addr["b"]), struct.pack("<Q", addr["c"])], "not inside a structure"),
        # a tensor in two parameters
        ([(8, 8)] * 4, lambda t, addr: [struct.pack("<Q", addr[n]) for n in "abca"], "is in kernel parameters 0 and 3"),
        # a tensor in no parameter
        ([(8, 8)] * 2, lambda t, addr: [struct.pack("<Q", addr[n]) for n in "ab"], "no kernel parameter holds the address of \\['c'\\]"),
        # the tensors in another order than the stub's
        ([(8, 8)] * 3, lambda t, addr: [struct.pack("<Q", addr[n]) for n in "bac"], "takes the tensors in the order \\['b', 'a', 'c'\\]"),
        # a descriptor whose address is not first
        ([(128, 64), (8, 8), (8, 8)], lambda t, addr: [bytes(8) + struct.pack("<Q", addr["a"]) + bytes(112), struct.pack("<Q", addr["b"]), struct.pack("<Q", addr["c"])], "not inside a structure"),
    ],
)
def test_parameters_tilefish_cannot_reproduce_are_refused(layout, make, message):
    with pytest.raises(TilefishBuildError, match=message):
        _cute.classify_params("k", TENSORS, layout, captures(make), ADDRESSES)


READS_PTX = """\
.visible .entry k(
	.param .align 4 .b8 k_param_0[12],
	.param .align 64 .b8 k_param_1[128],
	.param .u64 k_param_2,
	.param .align 8 .b8 k_param_3[8],
	.param .align 8 .b8 k_param_4[16]
)
{
	ld.param.v2.b8 	{%rs2, %rs3}, [k_param_0+8];
	ld.param.b32 	%r59, [k_param_0+4];
	ld.param.b32 	%r58, [k_param_0];
	mov.b64 	%rd1, k_param_1;
	ld.param.u64 	%rd2, [k_param_2];
	ld.param.v4.u16 	{%rs4, %rs5, %rs6, %rs7}, [k_param_3];
}
"""


def test_the_ptx_says_which_bytes_of_each_parameter_the_kernel_reads():
    reads = _cute.ptx_param_reads(READS_PTX, "k")
    assert reads == [frozenset(range(10)), None, frozenset(range(8)), frozenset(range(8)), frozenset()]


def test_the_padding_of_a_structure_is_not_compared_and_is_passed_as_zeros():
    # the fast divisor of a persistent tile scheduler: {divisor, multiplier, shift, shift} and 2 bytes of padding the host function leaves as they were
    tensors = [Arg(0, "a")]
    layout = [(8, 8), (12, 4)]
    caps = [[struct.pack("<Q", addr["a"]), bytes.fromhex("01000000010000000000") + bytes([t, 0x81])] for t, addr in enumerate(ADDRESSES)]
    addresses = [{"a": addr["a"]} for addr in ADDRESSES]
    with pytest.raises(TilefishBuildError, match="byte\\(s\\) \\[10\\] of kernel parameter 1"):
        _cute.classify_params("k", tensors, layout, caps, addresses)  # every byte read: the padding is taken for a value that varies
    params = _cute.classify_params("k", tensors, layout, caps, addresses, [frozenset(range(8)), frozenset(range(10))])
    assert params[1]["value"] == "01000000010000000000" + "0000"
    # an address the code does not read whole is no tensor's parameter: the bytes of it that it reads vary
    with pytest.raises(TilefishBuildError, match="byte\\(s\\) \\[3\\] of kernel parameter 0 \\(8 bytes\\) differ"):
        _cute.classify_params("k", tensors, layout, caps, addresses, [frozenset(range(4)), frozenset(range(10))])


def test_classify_needs_tensors_apart_and_at_new_addresses():
    same = [dict(ADDRESSES[0]) for _ in range(3)]
    with pytest.raises(TilefishBuildError, match="at the same address in two captures"):
        _cute.classify_params("k", TENSORS, [(8, 8)] * 3, captures(lambda t, addr: [bytes(8)] * 3), same)
    shared = [{"a": 16, "b": 16, "c": 32}, {"a": 48, "b": 48, "c": 64}, {"a": 80, "b": 80, "c": 96}]
    with pytest.raises(TilefishBuildError, match="two tensors share an address"):
        _cute.classify_params("k", TENSORS, [(8, 8)] * 3, captures(lambda t, addr: [bytes(8)] * 3), shared)
    with pytest.raises(TilefishBuildError, match="PTX entry declares"):
        _cute.classify_params("k", TENSORS, [(8, 8)] * 3, captures(lambda t, addr: [bytes(4)] * 3), ADDRESSES)


def capture(**kw):
    values = dict(name="kern", grid=(8, 8, 1), block=(128, 1, 1), shared=213248, cluster=(2, 1, 1), policy=1, params=())
    values.update(kw)
    return _cute.Capture(**values)


def test_the_launch_is_the_captured_one_and_the_grid_counts_clusters():
    grid, block, shared, cluster = _cute.launch_of([capture(), capture(), capture()], "kern", "gemm")
    assert (grid, block, shared, cluster) == ((4, 8, 1), (128, 1, 1), 213248, (2, 1, 1))
    assert _cute.grid_text(grid) == "(4, 8, 1)"
    assert _cute.launch_of([capture(cluster=(1, 1, 1), policy=0)] * 2, "kern", "k")[0] == (8, 8, 1)


@pytest.mark.parametrize(
    "caps, message",
    [
        ([capture(), capture(grid=(16, 8, 1))], "the captured launches differ"),
        ([capture(name="other")] * 2, "of the function 'other'"),
        ([capture(pdl=True)] * 2, "programmatic dependent launch"),
        ([capture(cooperative=True)] * 2, "a cooperative launch"),
        ([capture(cluster=(4, 4, 1))] * 2, "more than the portable 8"),
        ([capture(policy=2)] * 2, "the cluster scheduling policy 2"),
        ([capture(grid=(7, 8, 1))] * 2, "not a whole number of clusters"),
        ([capture(block=(2048, 1, 1))] * 2, "not a block a launch takes"),
    ],
)
def test_a_launch_no_host_makes_is_refused(caps, message):
    with pytest.raises(TilefishBuildError, match=message):
        _cute.launch_of(caps, "kern", "gemm")


PTX = """\
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-00000000
// Cuda compilation tools, release 13.4, V13.4.59
// Based on NVVM 23.0.0
//

.version 8.8
.target sm_103a
.address_size 64

.visible .entry kern(
	.param .align 32 .b8 kern_param_0[64],
	.param .align 64 .b8 kern_param_1[128],
	.param .u64 kern_param_2,
	.param .align 8 .b8 kern_param_3[8],
	.param .u32 kern_param_4
)
.reqntid 128, 1, 1
.reqnctapercluster 2, 1, 1
{
	ret;
}
"""


def test_the_ptx_says_the_parameters_the_block_the_cluster_and_the_nvvm():
    assert _cute.ptx_entry_params(PTX, "kern") == [(64, 32), (128, 64), (8, 8), (8, 8), (4, 4)]
    assert _cute.ptx_reqntid(PTX, "kern") == (128, 1, 1)
    assert _cute.ptx_cluster(PTX, "kern") == (2, 1, 1)
    assert _cute.ptx_tools(PTX) == "Cuda compilation tools, release 13.4, V13.4.59"
    assert _cute.ptx_reqntid(PTX.replace(".reqntid 128, 1, 1\n", ""), "kern") is None
    with pytest.raises(TilefishBuildError, match="has no entry point 'other'"):
        _cute.ptx_entry_params(PTX, "other")


@pytest.mark.parametrize("address", [0, 8, 1 << 57, -16, True])
def test_a_descriptor_holds_a_16_byte_aligned_address_below_2_to_57(address):
    with pytest.raises(ValueError):
        _cute.descriptor("00" * 128, address)


# -- the manifest ------------------------------------------------------------------------------------------------------------------------------

SYMBOL = "kernel_cutlass_gemm_0"


def fake_cute_build(spec, *, arch="sm_103", params=None, grid=(1, 2, 1), block=(128, 1, 1), shared=213248, cluster=(2, 1, 1), ptx=PTX, **kw):
    """A ``BuiltKernel`` of a CuTe DSL ``spec`` as ``_cute.build_cute`` makes one (around a small ELF file), its parameters a descriptor of ``a`` and
    ``b`` (templates as classify_params makes them), constant bytes, and a pointer to ``c``."""
    cubin = cubin_like(shared=None, symbol=SYMBOL, **arch_header(arch))
    if params is None:
        tensors = {a.name: a for a in spec.args}
        template = "00" * 8 + WORDS.hex() + "00" * 64

        def tensor(slot, name, kind):
            a = tensors[name]
            out = {"slot": slot, "index": a.index, "name": name}
            if kind == "tensormap":
                out.update(kind="tensormap", type="tensormap", ptx_type="b8[128]", size=128)
            else:
                out.update(kind="pointer", type="*fp16", ptx_type="u64", size=8)
            out.update(dtype=a.dtype, access=a.access, shape=list(a.shape_dims))
            if kind == "tensormap":
                out["template"] = template
            return out

        params = [
            tensor(0, "a", "tensormap"),
            tensor(1, "b", "tensormap"),
            {"slot": 2, "index": 2, "name": "param_2", "kind": "bytes", "type": "bytes", "ptx_type": "b8[4]", "size": 4, "value": "00040000"},
            tensor(3, "c", "pointer"),
        ]
    values = dict(
        spec=spec,
        arch=arch,
        cubin=cubin,
        ptx=ptx,
        symbol=SYMBOL,
        shared=shared,
        num_warps=0,
        sha256=hashlib.sha256(cubin).hexdigest(),
        sha256_code=code_digest(cubin),
        triton_version="",
        ptxas_version="Cuda compilation tools, release 12.9, V12.9.83",
        ptx_target=f"{arch}a",
        options={},
        launch=LaunchAttributes(0, 0, False, cluster, cluster[0] * cluster[1] * cluster[2], False),
        binary_digest=hashlib.sha256(ptx.encode()).hexdigest(),
        binary_build={"source": "cute", "cutlass_dsl": DSL, "options": spec.binary["options"]},
        cute={"abi": {"params": params, "scratch": [], "scratch_slots": 0}, "grid": list(grid), "block": list(block), "nvvm": "Cuda compilation tools, release 13.4, V13.4.59"},
    )
    values.update(kw)
    return BuiltKernel(**values)


def exported(tmp_path, op=None, **kw):
    op = op if op is not None else declare()
    built = [fake_cute_build(s, **kw) for s in compile_specs(op)]
    path = write_manifest(built, tmp_path / "manifest.json", cubins=True)
    return op, built, path


def test_the_entry_of_a_cute_kernel_has_the_compiled_launch(tmp_path):
    op, built, path = exported(tmp_path)
    doc = json.loads(path.read_text())
    e = next(e for e in doc["kernels"] if e["stem"] == "gemm.M128")
    assert e["kind"] == "binary"
    assert e["binary"] == {"source": "cute", "cutlass_dsl": DSL, "options": ""}
    assert e["toolchain"] == {"cutlass_dsl": DSL, "nvvm": "Cuda compilation tools, release 13.4, V13.4.59", "ptxas": "Cuda compilation tools, release 12.9, V12.9.83"}
    assert e["source_digest_kind"] == "tilefish.cute.v1" and e["source_digest"] == hashlib.sha256(PTX.encode()).hexdigest()
    assert (e["grid"], e["block"], e["shared"], e["symbol"], e["ptx_target"]) == ("(1, 2, 1)", [128, 1, 1], 213248, SYMBOL, "sm_103a")
    assert e["launch"]["cluster_dims"] == [2, 1, 1] and e["launch"]["num_ctas"] == 2
    assert [p["kind"] for p in e["abi"]["params"]] == ["tensormap", "tensormap", "bytes", "pointer"]
    assert e["abi"]["scratch"] == [] and e["abi"]["scratch_slots"] == 0
    assert "persistent" not in e and "options" not in e and "num_warps" not in e
    loaded = Manifest.load(path)
    assert loaded.digest == manifest_digest(doc)


def test_a_replay_takes_the_entry_of_the_ptx_the_dsl_makes_now(tmp_path):
    op, built, path = exported(tmp_path)
    manifest = Manifest.load(path)
    spec = compile_specs(op)[0]
    digest = hashlib.sha256(PTX.encode()).hexdigest()
    b = manifest.built_kernel(spec, "sm_103", cute_digest=digest)
    assert b.cute["grid"] == [1, 2, 1] and b.grid == (1, 2, 1) and b.block == (128, 1, 1)
    assert b.abi()["params"][0]["kind"] == "tensormap" and b.shared == 213248 and b.launch.cluster_dims == (2, 1, 1)
    assert manifest.built_kernel(spec, "sm_103", pinned=True).cute is not None  # pinned: the caller vouches for it
    with pytest.raises(TilefishManifestError, match="now compiles to PTX with sha256 " + "0" * 64):
        manifest.built_kernel(spec, "sm_103", cute_digest="0" * 64, pinned=True)  # a recomputed digest that differs is refused, pinned or not
    with pytest.raises(TilefishManifestError, match="cannot tell whether the binary .* the DSL is not here"):
        manifest.built_kernel(spec, "sm_103", cute_why="the DSL is not here")


def test_a_replay_refuses_an_entry_of_another_declaration_or_another_abi(tmp_path, monkeypatch):
    op, built, path = exported(tmp_path)
    manifest = Manifest.load(path)
    other = declare(cute_options="--opt-level 2")
    with pytest.raises(TilefishManifestError, match="first difference at declaration_digest"):
        manifest.built_kernel(compile_specs(other)[0], "sm_103", pinned=True)
    renamed = declare()
    spec = compile_specs(renamed)[0]
    doc = json.loads(path.read_text())
    e = next(e for e in doc["kernels"] if e["stem"] == "gemm.M128")
    e["abi"]["params"][3]["shape"] = [256, 64]
    path.write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match="is not a binary of the compile spec: parameter 'c' has"):
        Manifest.load(path).built_kernel(spec, "sm_103", pinned=True)


def mutate(path, change, stem="gemm.M128"):
    doc = json.loads(path.read_text())
    change(next(e for e in doc["kernels"] if e["stem"] == stem))
    path.write_text(json.dumps(doc))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda e: e["abi"]["params"][0].update(template="11" + "00" * 127), "bytes 0 to 7 .* must be zero"),
        (lambda e: e["abi"]["params"][0].update(template="00" * 64 + "ff" * 64), "bytes 0 to 7 .* must be zero"),
        (lambda e: e["abi"]["params"][0].update(template="00" * 127), "template must be 256 lowercase hex digits"),
        (lambda e: e["abi"]["params"][0].update(strides=["s", 1], offset="o"), "has a dtype, an access, a shape and a template, and no"),
        (lambda e: e["abi"]["params"][0].update(size=64), "passed as 128 bytes"),
        (lambda e: e["abi"]["params"][2].update(value="0004"), "value must be 8 lowercase hex digits"),
        (lambda e: e["abi"]["params"][2].update(dtype="f16"), "constant bytes: it has a value and no"),
        (lambda e: e["abi"]["params"][2].update(ptx_type="b8[8]"), "ptx_type 'b8\\[<size>\\]'"),
        (lambda e: e["abi"]["params"][3].update(strides=[64, 1], offset="o"), "a view or an alias: a CuTe DSL kernel takes dense tensors"),
        (lambda e: e["abi"]["params"].append({"slot": 4, "index": 4, "name": "n", "kind": "scalar", "type": "i32", "ptx_type": "u32", "size": 4, "dim": "M"}), "a CuTe DSL kernel takes pointers"),
        (lambda e: e.update(binary={"source": "cute", "cutlass_dsl": DSL}), "binary of a CuTe DSL kernel must be"),
        (lambda e: e.update(binary={"source": "cute", "cutlass_dsl": "four", "options": ""}), "binary of a CuTe DSL kernel must be"),
        (lambda e: e.update(toolchain={"cutlass_dsl": DSL, "ptxas": "x"}), "the toolchain of a CuTe DSL kernel is"),
        (lambda e: e["toolchain"].update(cutlass_dsl="4.8.0"), "toolchain.cutlass_dsl '4.8.0' is not binary.cutlass_dsl"),
        (lambda e: e.update(source_digest_kind="tilefish.binary.v1"), "with source_digest_kind 'tilefish.cute.v1'"),
        (lambda e: e.update(grid="(M, 2, 1)"), "the grid of a CuTe DSL kernel is integers"),
        (lambda e: e.update(ptx_target="sm_90a"), "ptx_target of a CuTe DSL kernel is the .target of its PTX"),
        (lambda e: e.update(persistent={"programs_per_sm": 1}, grid="(2,)"), "has no persistent grid"),
        (lambda e: e["launch"].update(cluster_dims=[4, 4, 1], num_ctas=16), "its launch attributes are the zero ones but cluster_dims"),
    ],
)
def test_the_readers_refuse_a_malformed_cute_entry(tmp_path, change, message):
    _, _, path = exported(tmp_path)
    mutate(path, change)
    with pytest.raises(TilefishManifestError, match=message):
        Manifest.load(path)


def test_the_kinds_of_a_cute_kernel_are_refused_on_any_other_entry(tmp_path):
    _, _, path = exported(tmp_path)

    def as_cubin(e):
        e["binary"] = {"source": "cubin"}
        e["toolchain"] = {}
        e["source_digest_kind"] = "tilefish.binary.v1"
        e["source_digest"] = e["sha256"]

    mutate(path, as_cubin)
    with pytest.raises(TilefishManifestError, match="parameter a is a tensormap, which is a parameter of a CuTe DSL kernel"):
        Manifest.load(path)


def test_a_merge_refuses_a_cute_entry_of_another_launch(tmp_path):
    op, built, path = exported(tmp_path)
    again = [fake_cute_build(s, grid=(2, 2, 1)) for s in compile_specs(op)]
    with pytest.raises(TilefishManifestError, match="already has a different binary for 'gemm.M128' on sm_103 \\(grid differs\\)"):
        write_manifest(again, path, cubins=True, merge=True)
    write_manifest(again, path, cubins=True, merge=True, overwrite=True)
    assert json.loads(path.read_text())["kernels"][0]["grid"] == "(2, 2, 1)"


def test_the_rust_and_cpp_emitters_type_a_cute_kernel_by_its_tensors(tmp_path):
    from tilefish.hosts import cpp, rust

    op, built, path = exported(tmp_path)
    manifest = Manifest.load(path, pin=Manifest.load(path).digest)
    text = rust.render(manifest)
    assert "pub struct GemmM128" in text and "The CuTe DSL kernel `gemm.M128`" in text
    assert "passed as the TMA descriptor the CuTe DSL encoded" in text
    assert "class GemmM128" in cpp.render(manifest) or "GemmM128" in cpp.render(manifest)


# -- the launch, through a driver that records it ----------------------------------------------------------------------------------------------


class _Result:
    CUDA_SUCCESS = 0


class _Attr:
    def __getattr__(self, name):
        return name


class FakeDriver:
    """Just enough of cuda.bindings.driver for LoadedKernel: it records the bytes of every parameter of a launch."""

    CUresult = _Result
    CUdevice_attribute = _Attr()
    CUfunction_attribute = _Attr()
    CUlaunchAttributeID = _Attr()
    CUclusterSchedulingPolicy = _Attr()

    def __init__(self, sizes):
        self.sizes = sizes
        self.launches = []

    def CUstream(self, s):
        return s

    def cuCtxGetCurrent(self):
        return (0, 1)

    def cuCtxGetDevice(self):
        return (0, 0)

    def cuDeviceGetAttribute(self, attr, device):
        return (0, 232448 if "SHARED" in attr else 1 << 30)

    def cuModuleLoadData(self, cubin):
        return (0, "module")

    def cuModuleGetFunction(self, module, symbol):
        return (0, "function")

    def cuFuncGetAttribute(self, attr, fn):
        return (0, 0)

    def cuFuncSetAttribute(self, fn, attr, value):
        return (0,)

    def cuFuncGetParamInfo(self, fn, index):
        if index < len(self.sizes):
            return (0, 0, self.sizes[index])
        return (1, 0, 0)

    def cuModuleUnload(self, module):
        return (0,)

    def _record(self, grid, block, shared, slots):
        array = ctypes.cast(slots, ctypes.POINTER(ctypes.c_void_p))
        params = [ctypes.string_at(array[i], size) for i, size in enumerate(self.sizes)]
        self.launches.append(dict(grid=grid, block=block, shared=shared, params=params, aligned=[array[i] % 64 for i in range(len(self.sizes))]))

    def cuLaunchKernel(self, fn, gx, gy, gz, bx, by, bz, shared, stream, slots, extra):
        self._record((gx, gy, gz), (bx, by, bz), shared, slots)
        return (0,)

    class _Config:
        pass

    class _LaunchAttribute:
        class _Value:
            class _Cluster:
                x = y = z = 1

            def __init__(self):
                self.clusterDim = self._Cluster()

        def __init__(self):
            self.value = self._Value()

    def CUlaunchConfig(self):
        return self._Config()

    def CUlaunchAttribute(self):
        return self._LaunchAttribute()

    def cuLaunchKernelEx(self, config, fn, slots, extra):
        self._record((config.gridDimX, config.gridDimY, config.gridDimZ), (config.blockDimX, config.blockDimY, config.blockDimZ), config.sharedMemBytes, slots)
        self.launches[-1]["cluster"] = (config.attrs[0].value.clusterDim.x, config.attrs[0].value.clusterDim.y, config.attrs[0].value.clusterDim.z)
        return (0,)


def test_a_launch_writes_each_address_into_its_template_and_passes_the_bytes(monkeypatch, tmp_path):
    op, built, path = exported(tmp_path)
    b = Manifest.load(path).built_kernel(compile_specs(op)[0], "sm_103", pinned=True)
    drv = FakeDriver([128, 128, 4, 8])
    monkeypatch.setattr(_cuda, "_driver", lambda: drv)
    monkeypatch.setattr(_cuda, "_loaded", {})
    loaded = _cuda.load(b, 0)
    from tilefish.runtime import cute_launch_args

    args = cute_launch_args(b, {"a": 0x7F0000100000, "b": 0x7F0000200000, "c": 0x7F0000300000})
    assert args == [("tensormap", 0x7F0000100000), ("tensormap", 0x7F0000200000), ("pointer", 0x7F0000300000)]
    loaded.launch(b.grid, args, stream=0)
    (launch,) = drv.launches
    assert launch["grid"] == (2, 2, 1) and launch["block"] == (128, 1, 1) and launch["shared"] == 213248 and launch["cluster"] == (2, 1, 1)
    assert launch["params"][0] == struct.pack("<Q", 0x7F0000100000) + WORDS + bytes(64)
    assert launch["params"][1][:8] == struct.pack("<Q", 0x7F0000200000)
    assert launch["params"][2] == bytes.fromhex("00040000")
    assert launch["params"][3] == struct.pack("<Q", 0x7F0000300000)
    assert launch["aligned"][0] == 0 and launch["aligned"][1] == 0  # a descriptor is 64-byte aligned, as its parameter is
    with pytest.raises(TilefishRuntimeError, match="16-byte aligned and below 2\\*\\*57"):
        loaded.launch(b.grid, [("tensormap", 0x7F0000100008), args[1], args[2]], stream=0)
    with pytest.raises(TilefishRuntimeError, match="takes 3 arguments"):
        loaded.launch(b.grid, args[:2], stream=0)
