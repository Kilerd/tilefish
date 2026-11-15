"""A lowered kernel given a view that does not start on 16 bytes runs its specialisation for it, instead of refusing it.

Every pointer of a lowered kernel is compiled as 16-byte aligned (``tt.divisibility=16``, a ``pointer_alignment`` of 16 bytes). ``kernel.unaligned("a")``
is the same kernel with ``a`` compiled aligned to its element only: no divisibility hint for it, a ``pointer_alignment`` of its element size, its own
operator name (``<kernel>__ua<indices>``), stems and declaration digest, the same generated source. A graph that passes a dense view at an offset that
is not provably a multiple of 16 bytes to a read-only parameter (a row of a ``[K, R]`` int32 plane with an odd ``R``) calls the specialisation, and
``tilefish.run`` of the kernel on a tensor whose address is not 16-byte aligned runs it. No GPU here: ``test_gpu_types.py`` runs them.
"""

import json
import pickle

import pytest
from built_fakes import fake_build
from fake_torch import FakeTensor, prepare, verdict

import tilefish
from tilefish import Dim, InOut, Manifest, Tensor, TilefishDeclarationError, TilefishPreconditionError, TilefishTypeError, f32, i32
from tilefish._build import spec_entry
from tilefish.aot import CompileSpec, compile_specs, declaration_digest, write_manifest
from tilefish.hosts import rust

R = Dim("R", buckets=[5, 8, 13])
R4 = Dim("R4", divisible_by=4)


@tilefish.kernel
def add_rows(a: Tensor[R, i32], b: Tensor[R, i32]) -> Tensor[R, i32]:
    return a + b


@tilefish.graph
def plane_sum(plane: Tensor[3, R, i32]):
    """Rows 1 and 2 of the plane start 4 * R and 8 * R bytes in, not provably a multiple of 16: two specialisations (types_kernels.py has the same)."""
    s = add_rows(plane[1], plane[0])
    return add_rows(s, plane[2])


def alignment(spec, name):
    return next(p.bytes for p in spec.preconditions if p.kind == "pointer_alignment" and p.arg == name)


def test_the_specialisation_is_an_operator_of_its_own():
    ua = add_rows.unaligned("a")
    assert ua.name == "add_rows__ua0" and ua.base is add_rows and ua.unaligned_params == frozenset({"a"})
    assert add_rows.unaligned("a") is ua and ua.unaligned() is ua and add_rows.unaligned() is add_rows
    both = add_rows.unaligned("a", "b")
    assert both.name == "add_rows__ua0_1" and ua.unaligned("b") is both and add_rows.unaligned("b", "a") is both
    assert ua.signature is add_rows.signature and ua.graph is add_rows.graph
    assert pickle.loads(pickle.dumps(ua)) is ua


def test_only_a_read_only_tensor_parameter_has_one():
    @tilefish.kernel
    def bump(x: Tensor[R, f32], acc: InOut[Tensor[R, f32]]):
        acc[...] = acc + x

    with pytest.raises(TilefishDeclarationError, match="is not a read-only tensor parameter"):
        bump.unaligned("acc")
    with pytest.raises(TilefishDeclarationError, match="is not a read-only tensor parameter"):
        add_rows.unaligned("c")
    bump.unaligned("x")


def test_the_source_is_the_kernels_and_the_binary_is_another():
    base, ua = add_rows.lower(), add_rows.unaligned("a").lower()
    assert ua.source == base.source and ua.fn_name == base.fn_name == "tf_add_rows" and ua.module == base.module
    assert ua.extern.name == "add_rows__ua0" and ua.source_sha256 == base.source_sha256
    assert declaration_digest(ua.extern) != declaration_digest(base.extern)
    (b,), (u,) = compile_specs(base.extern), compile_specs(ua.extern)
    assert u.stem == "add_rows__ua0" and b.stem == "add_rows"
    assert (0,) in b.attrs and (0,) not in u.attrs and u.attrs[(1,)] == b.attrs[(1,)]  # no divisibility hint for a, the same for b
    assert (alignment(b, "a"), alignment(u, "a"), alignment(u, "b"), alignment(u, "out")) == (16, 4, 16, 16)
    assert spec_entry(u)["preconditions"] != spec_entry(b)["preconditions"]
    back = CompileSpec.from_json(u.to_json())
    back.verify_against(ua.extern)
    assert back.verified
    with pytest.raises(tilefish.TilefishSpecError):
        CompileSpec.from_json(u.to_json()).verify_against(base.extern)


def test_a_dense_view_off_16_bytes_calls_the_specialisation():
    first, second = [n for n in plane_sum.graph.nodes if n.attrs.get("kind") == "kernel"]
    assert first.callee is add_rows.unaligned("a") and first.attrs["unaligned"] == ("a",)
    assert second.callee is add_rows.unaligned("b") and second.attrs["unaligned"] == ("b",)
    assert first.op == second.op == "add_rows"


def test_an_aligned_view_calls_the_kernel():
    @tilefish.graph
    def g(plane: Tensor[3, R4, i32]):  # R4 is a multiple of 4: a row of int32 starts on 16 bytes
        return add_rows(plane[1], plane[2])

    (node,) = [n for n in g.graph.nodes if n.attrs.get("kind") == "kernel"]
    assert node.callee is add_rows and "unaligned" not in node.attrs


def test_a_view_that_is_not_dense_is_still_refused():
    @tilefish.kernel
    def colsum(x: Tensor[R, f32], y: Tensor[R, f32]) -> Tensor[R, f32]:
        return x + y

    with pytest.raises(TilefishTypeError, match="is a strided view"):

        @tilefish.graph
        def g(plane: Tensor[R, 2, f32]):
            return colsum(plane.select(1, 0), plane.select(1, 1))  # columns: stride 2, not dense


def test_an_extern_still_takes_contiguous_tensors_only():
    n = Dim("n")

    @tilefish.extern(triton="m:k", grid=lambda n: (n,))
    def k(x: Tensor[n, i32], y: tilefish.Out[Tensor[n, i32]], n_elem: n): ...

    with pytest.raises(TilefishTypeError, match="is a strided view"):

        @tilefish.graph
        def g(plane: Tensor[3, n, i32]):
            return k(plane[1])


def test_the_plan_steps_name_the_specialisations():
    p = tilefish.plan(plane_sum)
    for r in (5, 8, 13):
        first, second = p.steps(R=r)
        assert first["entry"] == "add_rows__ua0" and second["entry"] == "add_rows__ua1"
        assert first["args"][0]["view"] == {"shape": [r], "strides": [1], "offset": r}
        assert second["args"][1]["view"] == {"shape": [r], "strides": [1], "offset": 2 * r}
    assert set(p.callees) == {"add_rows__ua0", "add_rows__ua1"}


def test_a_call_on_an_address_off_16_bytes_runs_the_specialisation():
    a = FakeTensor((5,), "i32", (1 << 44) + 20)  # 4-byte aligned, not 16
    b = FakeTensor((5,), "i32", 1 << 45)
    prepared, error = prepare(add_rows, [a, b], next_slab=1 << 46)
    assert error is None and prepared.spec.stem == "add_rows__ua0"
    assert verdict(prepared, error)["params"][:2] == [(1 << 44) + 20, 1 << 45]
    prepared, error = prepare(add_rows, [b, a], next_slab=1 << 46)
    assert error is None and prepared.spec.stem == "add_rows__ua1"
    prepared, error = prepare(add_rows, [a, a], next_slab=1 << 46)
    assert error is None and prepared.spec.stem == "add_rows__ua0_1"
    prepared, error = prepare(add_rows, [b, b], next_slab=1 << 46)
    assert error is None and prepared.spec.stem == "add_rows"
    # the binaries of the kernel itself refuse the address (what a host decides for that entry)
    prepared, error = prepare(add_rows, [a, b], next_slab=1 << 46, specialise=False)
    assert prepared is None and isinstance(error, TilefishPreconditionError) and error.kind == "pointer_alignment"
    # an address that is not even a multiple of the element is refused by the specialisation
    odd = FakeTensor((5,), "i32", (1 << 44) + 2)
    prepared, error = prepare(add_rows, [odd, b], next_slab=1 << 46)
    assert prepared is None and error.kind == "pointer_alignment"


def test_the_manifest_readers_take_an_element_aligned_dense_pointer(tmp_path):
    (spec,) = compile_specs(add_rows.unaligned("a").lower().extern)
    write_manifest([fake_build(spec, arch="sm_103")], tmp_path / "manifest.json", cubins=True)
    entry = json.loads((tmp_path / "manifest.json").read_text())["kernels"][0]
    assert {"kind": "pointer_alignment", "arg": "a", "index": 0, "bytes": 4} in entry["preconditions"]
    m = Manifest.load(tmp_path)
    m.built_kernel(spec, "sm_103", pinned=True)
    rust._validate_entry(entry)
    m.check_pin(m.digest)
    assert "add_rows__ua0" in rust.render(m)


def test_a_device_scalar_is_still_16_byte_aligned():
    @tilefish.kernel
    def scaled(x: Tensor[R, f32], s: tilefish.DevScalar[f32]) -> Tensor[R, f32]:
        return x * tilefish.load(s)

    (spec,) = compile_specs(scaled.lower().extern)
    doc = spec.to_dict()
    i = next(k for k, p in enumerate(doc["preconditions"]) if p["kind"] == "pointer_alignment" and p["arg"] == "s")
    doc["preconditions"][i]["bytes"] = 4
    del doc["attrs"][str(doc["preconditions"][i]["index"])]
    with pytest.raises(tilefish.TilefishSpecError, match="pointer_alignment of the device scalar s must be 16 bytes"):
        CompileSpec.from_dict(doc)
