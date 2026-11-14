"""TMA tensor descriptors (``tilefish.TensorDesc``), with no GPU: the declaration, the compile spec, the manifest ABI and its readers, the launch
checks and the arguments of the encode, the plan runner on a fake backend, and the Rust emitter. ``test_gpu_tma.py`` runs the kernels.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import json

import pytest
from built_fakes import fake_build
from fake_torch import FakeTensor, prepare

import tilefish
from tilefish import (
    DevScalar,
    Dim,
    InOut,
    MayAlias,
    Out,
    StrideOf,
    Strided,
    Tensor,
    TensorDesc,
    TilefishBuildError,
    TilefishDeclarationError,
    TilefishManifestError,
    TilefishSpecError,
    bf16,
    const,
    f16,
    f32,
    i32,
)
from tilefish import _tensormap
from tilefish._build import abi_of, abi_slot_sizes, check_tensordesc_meta, ptx_params, write_manifest
from tilefish._launch import TensorInfo
from tilefish.aot import CompileSpec, compile_specs, declaration_digest
from tilefish.runtime import tensor_map_arg

R, C, K = Dim("R"), Dim("C"), Dim("K")
Rb = Dim("Rb", buckets=[64, 192])


def tile_copy(language="triton", **map_kw):
    """A tiled copy through two descriptors of [64, 64] f16 blocks (by reference: nothing is imported)."""
    kernel = {language: f"tma_mod:tile_copy_{language}"}

    @tilefish.extern(**kernel, num_warps=4, grid=lambda R, C: (tilefish.cdiv(R, 64), tilefish.cdiv(C, 64)), name=f"tile_copy_{language}")
    def op(
        x: TensorDesc(Tensor[R, C, f16], block=(64, 64), **map_kw),
        y: Out[TensorDesc(Tensor[R, C, f16], block=(64, 64), **map_kw)],
        BM: const = 64,
        BN: const = 64,
    ): ...

    return op


@tilefish.extern(triton="tma_mod:mm", num_warps=4, grid=lambda R, C: (tilefish.cdiv(R, 128), tilefish.cdiv(C, 128)))
def mm(
    a: TensorDesc(Tensor[R, K, f16], block=(128, 64)),
    b: TensorDesc(Tensor[C, K, f16], block=(128, 64)),
    c: Out[TensorDesc(Tensor[R, C, f16], block=(128, 128))],
    k_len: K,
    BM: const = 128,
): ...


@tilefish.extern(triton="tma_mod:view_copy", num_warps=4, grid=lambda R, C: (tilefish.cdiv(R, 64), tilefish.cdiv(C, 64)))
def view_copy(x: TensorDesc(Strided[Tensor[R, C, bf16]], block=(64, 32)), y: Out[Tensor[R, C, bf16]], BM: const = 64): ...


# -- the declaration --------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block, bits, swizzle, box",
    [
        ((64, 64), 16, 128, (64, 64)),
        ((128, 128), 16, 128, (128, 64)),  # rows wider than the span: two boxes per block (Triton 3.5.1 records [128, 64])
        ((16, 64), 8, 64, (16, 64)),
        ((32, 64), 32, 128, (32, 32)),
        ((16, 256), 8, 128, (16, 128)),
        ((64, 32), 16, 64, (64, 32)),
        ((64, 16), 16, 32, (64, 16)),
        ((64, 8), 16, 0, (64, 8)),  # 16-byte rows: no swizzle
        ((4, 64), 16, 0, (4, 64)),  # fewer than 8 rows: no swizzle
        ((256,), 16, 0, (256,)),  # rank 1: no swizzle
    ],
)
def test_the_default_swizzle_and_the_box_are_what_triton_records(block, bits, swizzle, box):
    """The swizzle a declaration gets by default and the box of its map are what Triton 3.5.1's ``tensordesc_meta`` said for these blocks on a
    B300 (each probed with ``triton.compile``); the build checks them against the compiled kernel either way."""
    assert _tensormap.default_swizzle(block, bits) == swizzle
    assert _tensormap.box_of(block, swizzle, bits) == box


def test_a_descriptor_is_a_tensor_parameter_with_its_map():
    op = tile_copy()
    x, y = op.signature.params[:2]
    assert (x.kind, x.access, y.access) == ("tensor", "in", "out")
    assert x.tensor_map == _tensormap.TensorMapSpec((64, 64), (64, 64), 128, 128, "zero")
    assert str(op.signature).startswith("(x: TensorDesc(Tensor[R, C, f16], block=(64, 64), swizzle=128, l2_promotion=128, oob_fill='zero'), y: Out[TensorDesc(")
    (x,) = view_copy.signature.params[:1]
    assert x.strided and x.align == 16 and [str(s) for s in x.type.strides] == ["x_s0", "1"] and str(x.type.offset) == "x_offset"
    assert x.type.offset.divisible_by == 8  # a 16-byte aligned start: the global address of a map


@pytest.mark.parametrize(
    "make, why",
    [
        (lambda: TensorDesc(Tensor[R, 64, f16], block=(512, 64), swizzle=0), "1 to 256"),
        (lambda: TensorDesc(Tensor[R, 64, f16], block=(64, 4), swizzle=0), "multiple of 16"),
        (lambda: TensorDesc(Tensor[R, 64, f16], block=(64, 48)), "power of two"),
        (lambda: TensorDesc(Tensor[R, 64, f16], block=(64,)), "one int per axis"),
        (lambda: TensorDesc(Tensor[R, 64, f16], block=(64, 64), swizzle=16), "one of"),
        (lambda: TensorDesc(Tensor[R, 64, f16], block=(64, 64), l2_promotion=32), "l2_promotion"),
        (lambda: TensorDesc(Tensor[R, 64, f16], block=(64, 64), oob_fill="one"), "'zero' or 'nan'"),
        (lambda: TensorDesc(Tensor[R, 64, i32], block=(64, 64), oob_fill="nan"), "float NaN"),
        (lambda: TensorDesc(Tensor[R, 2, 2, 2, 2, 2, f16], block=(1, 1, 1, 1, 1, 8)), "rank 1 to 5"),
        (lambda: TensorDesc(DevScalar[i32], block=()), "plain tensor"),
        (lambda: TensorDesc(f16, block=(8,)), "describes a tensor"),
        (lambda: Out[TensorDesc(Strided[Tensor[R, 64, f16]], block=(64, 64))], "only a descriptor the kernel reads"),
        (lambda: Out[TensorDesc(Tensor[R, 64, f16], block=(64, 64)), MayAlias("x")], "may not be written over an input"),
    ],
)
def test_a_descriptor_the_driver_or_triton_refuses_is_refused_where_it_is_declared(make, why):
    with pytest.raises(TilefishDeclarationError, match=why):
        make()


def test_only_the_extern_of_a_triton_or_gluon_kernel_takes_a_descriptor(tmp_path):
    with pytest.raises(TilefishDeclarationError, match="only the extern of a Triton or Gluon kernel"):

        @tilefish.kernel
        def k(x: TensorDesc(Tensor[R, 64, f16], block=(64, 64))) -> Tensor[R, 64, f16]:
            return x

    ptx = tmp_path / "k.ptx"
    ptx.write_text(".version 8.8\n.target sm_90a\n.address_size 64\n.visible .entry k(.param .u64 k_param_0)\n{\nret;\n}\n")
    with pytest.raises(TilefishDeclarationError, match="only the extern of a Triton or Gluon kernel"):

        @tilefish.extern(ptx=str(ptx), symbol="k", block=(128,), grid=lambda R: (R,))
        def prebuilt(x: TensorDesc(Tensor[R, 64, f16], block=(64, 64))): ...

    with pytest.raises(TilefishDeclarationError, match="passed in its own slots"):

        @tilefish.extern(triton="tma_mod:k", grid=lambda R: (R,))
        def strided(x: TensorDesc(Strided[Tensor[R, 64, f16]], block=(64, 64)), s: StrideOf("x", 0)): ...

    with pytest.raises(TilefishDeclarationError, match="a tensor descriptor"):

        @tilefish.extern(triton="tma_mod:k", grid=lambda R: (R,))
        def aliased(x: TensorDesc(Tensor[R, 64, f32], block=(64, 64)), y: Out[Tensor[R, 64, f32], MayAlias("x")]): ...

    with pytest.raises(TilefishDeclarationError, match="clashes with the stride dimension"):

        @tilefish.extern(triton="tma_mod:k", grid=lambda R: (R,))
        def clash(x: TensorDesc(Strided[Tensor[R, 64, f16]], block=(64, 64)), x_s0: R): ...


def test_an_inout_descriptor_is_read_and_written():
    @tilefish.extern(triton="tma_mod:bump", grid=lambda R: (R,))
    def bump(x: InOut[TensorDesc(Tensor[R, 64, f32], block=(64, 64))]): ...

    (spec,) = compile_specs(bump)
    (a,) = spec.args
    assert (a.kind, a.access) == ("tensordesc", "inout")


# -- the compile spec and the declaration digest ------------------------------------------------------------------------------------------------


def test_the_compile_spec_has_tritons_signature_of_each_descriptor():
    (spec,) = compile_specs(tile_copy())
    assert spec.signature == {"x": "tensordesc<fp16[64, 64]>", "y": "tensordesc<fp16[64, 64]>", "BM": "constexpr", "BN": "constexpr"}
    assert spec.attrs == {}  # a descriptor takes no tt.divisibility
    x = spec.args[0]
    assert (x.kind, x.dtype, x.shape_dims, x.access) == ("tensordesc", "f16", ("R", "C"), "in")
    assert x.tensor_map == _tensormap.TensorMapSpec((64, 64), (64, 64), 128).to_json("f16")
    pres = [p.to_dict() for p in spec.preconditions]
    assert pres == [
        {"kind": "pointer_alignment", "arg": "x", "index": 0, "bytes": 16},
        {"kind": "pointer_alignment", "arg": "y", "index": 1, "bytes": 16},
    ]  # no max_elements: the kernel addresses a descriptor's tensor by coordinates
    assert spec.schema == 3
    (g,) = compile_specs(tile_copy("gluon"))
    assert g.signature["x"] == (
        "tensordesc<fp16[64, 64],NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2, transposed=False, fp4_padded=False, "
        "ctas_per_cga=[1, 1], cta_split_num=[1, 1], cta_order=[1, 0])>"
    )  # Gluon's own TensorDescriptor spells its layout so: the frontend takes the layout from the type


def test_a_spec_round_trips_and_an_edited_descriptor_is_refused():
    for op in (tile_copy(), tile_copy("gluon"), mm, view_copy):
        (spec,) = compile_specs(op)
        back = CompileSpec.from_json(spec.to_json())
        back.verify_against(op)
        assert back.verified and back.to_dict() == spec.to_dict()
    (spec,) = compile_specs(tile_copy())
    doc = spec.to_dict()
    doc["args"][0]["tensor_map"]["box"] = [64, 32]
    edited = CompileSpec.from_dict(doc)  # well formed (a box that tiles the block)...
    with pytest.raises(TilefishSpecError, match="tensor_map"):
        edited.verify_against(tile_copy())  # ...but not the declaration's
    for change, why in (
        (lambda d: d["args"][0]["tensor_map"].update(box=[512, 64]), "1 to 256"),
        (lambda d: d["args"][0]["tensor_map"].update(swizzle="CU_TENSOR_MAP_SWIZZLE_64B"), "swizzle span"),
        (lambda d: d["args"][0]["tensor_map"].update(data_type="CU_TENSOR_MAP_DATA_TYPE_FLOAT32"), "data_type"),
        (lambda d: d["args"][0]["tensor_map"].pop("interleave"), "exactly the fields"),
        (lambda d: d["args"][0].update(type="tensordesc<fp16[64, 32]>"), "signature type"),
        (lambda d: d["preconditions"][0].update(bytes=2), "16 bytes"),
    ):
        doc = spec.to_dict()
        change(doc)
        doc["signature"] = {a["name"]: a["type"] for a in doc["args"]}
        doc["launch_params"] = [{"name": a["name"], "type": a["type"]} for a in doc["args"] if a["kind"] != "constexpr"] + doc["launch_params"][2:]
        with pytest.raises(TilefishSpecError, match=why):
            CompileSpec.from_dict(doc)


def test_the_declaration_digest_has_the_map():
    base = declaration_digest(tile_copy())
    assert declaration_digest(tile_copy()) == base
    for kw in ({"swizzle": 64}, {"l2_promotion": 256}, {"oob_fill": "nan"}):
        assert declaration_digest(tile_copy(**kw)) != base, kw
    assert declaration_digest(tile_copy("gluon")) != base


# -- the ABI of the manifest ------------------------------------------------------------------------------------------------------------------


def test_a_descriptor_takes_one_plus_two_rank_slots_of_the_abi():
    (spec,) = compile_specs(mm)
    abi = abi_of(spec)
    assert [(p["name"], p["kind"], p["slot"], p["type"], p["ptx_type"], p["size"]) for p in abi["params"]] == [
        ("a", "tensordesc", 0, "tensordesc", "b8[128]", 128),
        ("b", "tensordesc", 5, "tensordesc", "b8[128]", 128),
        ("c", "tensordesc", 10, "tensordesc", "b8[128]", 128),
        ("k_len", "scalar", 15, "i32", "u32", 4),
    ]
    assert [s["slot"] for s in abi["scratch"]] == [16, 17]
    assert abi_slot_sizes(abi) == [128, 4, 4, 8, 8] * 3 + [4, 8, 8]
    c = abi["params"][2]
    assert (c["access"], c["shape"], c["tensor_map"]["block"], c["tensor_map"]["box"]) == ("out", ["R", "C"], [128, 128], [128, 64])
    (spec,) = compile_specs(view_copy)
    x = abi_of(spec)["params"][0]
    assert (x["strides"], x["offset"]) == (["x_s0", 1], "x_offset")


def test_the_ptx_of_a_descriptor_is_its_map_then_its_shape_and_strides():
    text = """(
        .param .align 64 .b8 k_param_0[128],
        .param .u32 k_param_1,
        .param .u32 k_param_2,
        .param .u64 k_param_3,
        .param .u64 k_param_4,
        .param .u64 .ptr .global .align 1 k_param_5,
        .param .u64 .ptr .global .align 1 k_param_6
    )"""
    assert ptx_params(text[1:]) == ["b8[128]", "u32", "u32", "u64", "u64", "u64", "u64"]


def test_the_build_refuses_a_kernel_that_expects_another_map():
    (spec,) = compile_specs(tile_copy())
    meta = {"swizzle": 3, "elem_size": 2, "elem_type": 6, "block_size": [64, 64], "fp4_padded": False}
    check_tensordesc_meta(spec, [meta, meta])
    with pytest.raises(TilefishBuildError, match="declare swizzle=64"):
        check_tensordesc_meta(spec, [{**meta, "swizzle": 2, "block_size": [64, 32]}, meta])
    with pytest.raises(TilefishBuildError, match="expects"):
        check_tensordesc_meta(spec, [meta, {**meta, "elem_type": 10}])
    with pytest.raises(TilefishBuildError, match="sm_90 or newer"):
        check_tensordesc_meta(spec, [])


def written(tmp_path, op=None, arch="sm_103"):
    (spec,) = compile_specs(op or mm)
    path = write_manifest([fake_build(spec, arch=arch)], tmp_path / "manifest.json", cubins=True)
    return spec, path


def test_the_manifest_reader_takes_a_descriptor_and_compares_it_with_the_spec(tmp_path):
    spec, path = written(tmp_path)
    m = tilefish.Manifest.load(tmp_path)
    entry = m.get(spec.stem, "sm_103").entry
    assert [p["kind"] for p in entry["abi"]["params"]] == ["tensordesc"] * 3 + ["scalar"]
    built = m.built_kernel(spec, "sm_103", pinned=True)
    assert built.abi() == abi_of(spec)


@pytest.mark.parametrize(
    "change, why",
    [
        (lambda p: p[0].pop("tensor_map"), "tensor_map must be an object"),
        (lambda p: p[0]["tensor_map"].update(box=[512, 64]), "1 to 256"),
        (lambda p: p[0]["tensor_map"].update(box=[48, 64]), "does not tile"),
        (lambda p: p[0]["tensor_map"].update(block=[128, 48]), "power of two"),
        (lambda p: p[0]["tensor_map"].update(swizzle="CU_TENSOR_MAP_SWIZZLE_32B"), "swizzle span"),
        (lambda p: p[0]["tensor_map"].update(oob_fill="CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA", data_type="CU_TENSOR_MAP_DATA_TYPE_INT32"), "data_type"),
        (lambda p: p[0]["tensor_map"].update(element_strides=[1, 2]), "element_strides"),
        (lambda p: p[0]["tensor_map"].update(interleave="CU_TENSOR_MAP_INTERLEAVE_32B"), "interleave"),
        (lambda p: p[0].update(ptx_type="u64", size=8), "128 bytes of a CUtensorMap"),
        (lambda p: p[0].update(type="*fp16"), "128 bytes of a CUtensorMap"),
        (lambda p: p[0].update(alias="b"), "no \\['alias'\\]"),
        (lambda p: p[0].update(strides=["a_s0", 1], offset="a_offset"), "only a descriptor the kernel reads|dims does not name"),
        (lambda p: p[2].update(strides=["c_s0", 1], offset="c_offset"), "only a descriptor the kernel reads"),
        (lambda p: p[1].update(slot=1), "its slots begin at 5"),
        (lambda p: p[3].update(tensor_map=p[0]["tensor_map"]), "only a tensor descriptor"),
        (lambda p: p[0].update(shape=["R", "K", 1, 1, 1, 1]), "1 to 5 axes"),
    ],
)
def test_the_manifest_reader_refuses_a_descriptor_that_is_not_what_it_says(tmp_path, change, why):
    spec, path = written(tmp_path)
    doc = json.loads(path.read_text())
    change(doc["kernels"][0]["abi"]["params"])
    path.write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError, match=why):
        tilefish.Manifest.load(tmp_path)


def test_a_max_elements_precondition_is_not_one_of_a_descriptor(tmp_path):
    spec, path = written(tmp_path)
    doc = json.loads(path.read_text())
    doc["kernels"][0]["preconditions"].append({"kind": "max_elements", "arg": "a", "index": 0, "value": 2147483647, "shape": ["R", "K"]})
    path.write_text(json.dumps(doc))
    with pytest.raises(TilefishManifestError):
        tilefish.Manifest.load(tmp_path).built_kernel(spec, "sm_103", pinned=True)


# -- a launch: the checks and the arguments of the encode ------------------------------------------------------------------------------------


SLAB = 1 << 40


def test_a_call_binds_a_descriptors_tensor_as_a_pointers():
    a, b = FakeTensor((200, 72), "f16", SLAB), FakeTensor((136, 72), "f16", 2 * SLAB)
    prepared, error = prepare(mm, [a, b], next_slab=4 * SLAB)
    assert error is None, error
    assert prepared.grid == (2, 2) and prepared.scalars == {"k_len": 72}
    assert prepared.tensors["c"].shape == (200, 136)


@pytest.mark.parametrize(
    "a, kind",
    [
        (FakeTensor((200, 72), "f16", SLAB + 2), "pointer_alignment"),  # 2 bytes past a 16-byte boundary
        (FakeTensor((200, 36), "f16", SLAB), "tensor_map"),  # rows of 72 bytes
        (FakeTensor((200, 72), "f16", SLAB, (80, 1)), "contiguous"),  # a dense descriptor takes a contiguous tensor
        (FakeTensor((200, 72), "f32", SLAB), "dtype"),
    ],
)
def test_a_tensor_the_driver_could_not_describe_is_refused_with_a_kind(a, kind):
    b = FakeTensor((136, a.shape[1]), "f16", 2 * SLAB)
    prepared, error = prepare(mm, [a, b], next_slab=4 * SLAB)
    assert prepared is None and error.kind == kind, error


def test_a_dimension_beyond_i32_is_refused_with_the_kind_tensor_map():
    big, b = FakeTensor((1 << 31, 64), "f16", SLAB), FakeTensor((128, 64), "f16", 2 * SLAB)
    prepared, error = prepare(mm, [big, b], next_slab=4 * SLAB)
    assert prepared is None and error.kind == "tensor_map" and "i32" in str(error)


@pytest.mark.parametrize(
    "strides, offset_bytes, kind",
    [
        ((256, 1), 0, None),
        ((196, 1), 0, "tensor_map"),  # a row stride of 392 bytes
        ((256, 1), 8, "pointer_alignment"),
        ((256, 2), 0, "stride"),  # the last axis is not contiguous
        ((0, 1), 0, None),  # an expanded view: stride 0 is a multiple of 16 bytes (the driver takes it)
    ],
)
def test_a_descriptor_of_a_view_holds_its_strides_to_the_rules(strides, offset_bytes, kind):
    x = FakeTensor((96, 192), "bf16", SLAB + offset_bytes, strides)
    prepared, error = prepare(view_copy, [x], next_slab=4 * SLAB)
    if kind is None:
        assert error is None, error
    else:
        assert prepared is None and error.kind == kind, error


def test_the_encode_arguments_are_tritons():
    (spec,) = compile_specs(mm)
    seen = []

    def encode(arguments):
        seen.append(arguments)
        return bytes(range(128))

    info = TensorInfo(shape=(200, 72), dtype="torch.float16", ptr=SLAB, contiguous=True, device="cuda:0", nbytes=200 * 72 * 2, strides=(72, 1), itemsize=2)
    arg = tensor_map_arg(spec, "a", info, encode)
    assert arg == _tensormap.TensorMapArg(bytes(range(128)), (200, 72), (72, 1))
    (e,) = seen
    assert e == _tensormap.EncodeArguments(
        data_type=6,
        rank=2,
        address=SLAB,
        global_dim=(72, 200),
        global_strides=(144, 200 * 144),  # the row in bytes, then the whole tensor (Triton's fillTMADescriptor writes it there)
        box_dim=(64, 128),
        element_strides=(1, 1),
        interleave=0,
        swizzle=3,
        l2_promotion=2,
        oob_fill=0,
    )
    one = _tensormap.encode_arguments("bf16", _tensormap.TensorMapSpec((256,), (256,), 0), SLAB, (1000,), (1,))
    assert (one.global_dim, one.global_strides, one.box_dim) == ((1000,), (2000,), (256,))
    three = _tensormap.encode_arguments("f32", _tensormap.TensorMapSpec((2, 8, 32), (2, 8, 32), 128), SLAB, (5, 6, 40), (480, 80, 1))
    assert (three.data_type, three.global_dim, three.global_strides) == (7, (40, 6, 5), (320, 1920, 9600))


# -- a plan: every step's descriptors are encoded from what it binds, before the first launch ----------------------------------------------


@tilefish.kernel
def doubled(x: Tensor[Rb, 128, f16]) -> Tensor[Rb, 128, f16]:
    return (x.to(f32) * 2.0).to(f16)


@tilefish.extern(triton="tma_mod:copy", num_warps=4, grid=lambda Rb: (tilefish.cdiv(Rb, 64),))
def copy128(x: TensorDesc(Tensor[Rb, 128, f16], block=(64, 64)), y: Out[TensorDesc(Tensor[Rb, 128, f16], block=(64, 64))]): ...


@tilefish.graph
def chain(x: Tensor[Rb, 128, f16]) -> Tensor[Rb, 128, f16]:
    return copy128(doubled(x))


class EncodingBackend:
    """The fake backend of ``test_planrun`` that also encodes tensor maps (bytes made of the address), or refuses one."""

    def __new__(cls, refuse=False):
        from test_planrun import FakeBackend

        class Backend(FakeBackend):
            def encode_tensor_map(self, arguments):
                self.log.append(("encode", arguments.address, arguments.global_dim))
                if refuse:
                    raise tilefish.TilefishRuntimeError("CUDA driver call cuTensorMapEncodeTiled failed")
                return arguments.address.to_bytes(8, "little") * 16

        return Backend()


def test_a_plan_steps_descriptors_are_encoded_from_its_buffers_and_launched_by_value():
    from tilefish._planrun import PlanRunner
    from test_planrun import ARENA, FakeProvider

    plan = tilefish.plan(chain)
    be = EncodingBackend()
    runner = PlanRunner(plan, FakeProvider(), be)
    x = FakeTensor((64, 128), "f16", 1 << 36)
    out = FakeTensor((64, 128), "f16", 1 << 37)
    runner.run((x,), {}, outputs=out)
    (t0,) = [b for b in plan.buffers if b["role"] == "intermediate"]
    encodes = [e for e in be.log if e[0] == "encode"]
    assert encodes == [("encode", ARENA + t0["offset"], (128, 64)), ("encode", 1 << 37, (128, 64))]
    first_launch = next(i for i, e in enumerate(be.log) if e[0] == "launch")
    assert max(i for i, e in enumerate(be.log) if e[0] == "encode") < first_launch, "every map is encoded before the first launch"
    copy_launch = [e for e in be.log if e[0] == "launch"][-1]
    (kind_x, arg_x), (kind_y, arg_y) = copy_launch[3]
    assert (kind_x, kind_y) == ("tensordesc", "tensordesc")
    assert arg_x == _tensormap.TensorMapArg((ARENA + t0["offset"]).to_bytes(8, "little") * 16, (64, 128), (128, 1))
    assert arg_y.descriptor == (1 << 37).to_bytes(8, "little") * 16
    # a capture holds the maps of the addresses it binds, encoded before it begins
    be.log.clear()
    runner.capture((x,), {}, out, None)
    assert [e[0] for e in be.log].index("begin") > max(i for i, e in enumerate(be.log) if e[0] == "encode")


def test_a_map_the_driver_refuses_launches_nothing():
    from tilefish._planrun import PlanRunner
    from test_planrun import FakeProvider

    be = EncodingBackend(refuse=True)
    runner = PlanRunner(tilefish.plan(chain), FakeProvider(), be)
    with pytest.raises(tilefish.TilefishRuntimeError, match="cuTensorMapEncodeTiled"):
        runner.run((FakeTensor((64, 128), "f16", 1 << 36),), {}, outputs=FakeTensor((64, 128), "f16", 1 << 37))
    assert not [e for e in be.log if e[0] in ("launch", "devscalar", "wait")], "nothing was enqueued"


def test_a_plans_steps_bind_descriptors_as_pointers():
    plan = tilefish.plan(chain)
    step = plan.steps(Rb=64)[1]
    assert step["entry"] == "copy128" and [a["name"] for a in step["args"]] == ["x", "y"]


# -- the Rust emitter -------------------------------------------------------------------------------------------------------------------------


def test_the_rust_launcher_of_a_descriptor_takes_its_tensor(tmp_path):
    from tilefish.hosts import rust

    spec, path = written(tmp_path)
    m = tilefish.Manifest.load(tmp_path, pin=tilefish.Manifest.load(tmp_path).digest)
    out = rust.emit(m, tmp_path / "launchers.rs", ops=[mm])
    text = out.read_text()
    assert "a: TensorRef<'a, F16>" in text and "c: TensorMut<'a, F16>" in text
    assert "passed as a TMA descriptor (box `[128, 64]`, SWIZZLE_128B, L2_PROMOTION_L2_128B, FLOAT_OOB_FILL_NONE)" in text
    assert "encodes its `CUtensorMap`" in text


def test_a_dimension_beyond_i32_and_a_last_stride_are_checked_by_the_module_too():
    with pytest.raises(tilefish.TilefishPreconditionError) as e:
        _tensormap.check_tensor_map("k", "x", "f16", (1 << 31,), (1,))
    assert e.value.kind == "tensor_map"
    with pytest.raises(tilefish.TilefishPreconditionError) as e:
        _tensormap.check_tensor_map("k", "x", "f16", (8, 8), (8, 2))
    assert e.value.kind == "tensor_map" and "last axis" in str(e.value)
    with pytest.raises(tilefish.TilefishPreconditionError) as e:
        _tensormap.check_tensor_map("k", "x", "u8", (2, 8), (1 << 40, 1))
    assert e.value.kind == "tensor_map"
    _tensormap.check_tensor_map("k", "x", "u8", (2, 8), (16, 1))
