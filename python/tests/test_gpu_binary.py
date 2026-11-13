"""Prebuilt binaries that Triton does not compile, on a real GPU: PTX that ptxas assembles and cubins that are used as they are.

``@tilefish.extern(ptx="k.ptx", symbol=..., block=..., grid=...)`` declares a kernel by its C ABI: its pointers and scalars in order, and no Triton
scratch slots. Three things are checked here.

* **The same kernel through both doors.** The PTX of a lowered golden (the elementwise gelu and the row softmax) is taken from a Triton build and
  declared as a prebuilt extern with the Triton ABI's parameters and its two scratch slots written out as explicit fixed arguments: its output is
  byte-identical to the Triton path's, in ``jit`` and in ``replay``.
* **Hand-written PTX** (``ptx/axpy_f32.ptx``, a fused multiply-add with a float constant argument; ``ptx/strided_copy_bf16.ptx``, a copy of a strided
  view with a 64-bit row stride and an integer constant) is correct, and a replay of its manifest is byte-identical to ``jit``.
* **A cubin is used as it is** for its architecture (made here from the PTX with ptxas), and refused for another.
"""

import hashlib
import pathlib

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

from gpu_util import randn, same_bytes  # noqa: E402
from lower_cases import T, gelu_exact, softmax  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Out, StrideOf, Strided, Tensor, TilefishBuildError, TilefishManifestError, bf16, const, f32, i32, i64, runtime  # noqa: E402
from tilefish import _binary  # noqa: E402
from tilefish._build import build  # noqa: E402
from tilefish.aot import compile_specs  # noqa: E402

PTX = pathlib.Path(__file__).resolve().parent / "ptx"
n = Dim("n")
R = Dim("R")


def capability():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


# -- hand-written PTX -------------------------------------------------------------------------------------------------------------------


@tilefish.extern(ptx=PTX / "axpy_f32.ptx", symbol="axpy_f32", block=(128,), grid=lambda n: (tilefish.cdiv(n, 128),))
def axpy(x: Tensor[n, f32], y: Tensor[n, f32], out: Out[Tensor[n, f32]], n_elem: n, alpha: const[f32] = 2.0): ...


@tilefish.extern(
    ptx=PTX / "strided_copy_bf16.ptx", symbol="strided_copy_bf16", block=(64,), grid=lambda R: (R,), index_bits=64
)
def strided_copy(x: Strided[Tensor[R, 64, bf16]], y: Out[Tensor[R, 64, bf16]], *, cols: const[i32] = 64, x_s0: StrideOf("x", 0)): ...


@pytest.mark.parametrize("size", [1, 17, 1000, 4096, 100003])
def test_hand_written_axpy_is_correct(size):
    x, y = randn((size,), torch.float32, seed=size), randn((size,), torch.float32, seed=size + 1)
    got = tilefish.run(axpy, x, y)
    assert same_bytes(got, 2.0 * x + y)  # 2 * x is exact, so the fused and the separate roundings agree


def test_a_fixed_argument_is_in_the_manifest_and_cannot_be_changed_at_a_call(tmp_path):
    runtime.export(axpy, tmp_path)
    import json

    entry = json.loads((tmp_path / "manifest.json").read_text())["kernels"][0]
    assert entry["kind"] == "binary" and entry["abi"]["scratch_slots"] == 0 and entry["abi"]["scratch"] == []
    alpha = entry["abi"]["params"][-1]
    assert (alpha["kind"], alpha["type"], alpha["ptx_type"], alpha["size"], alpha["value"]) == ("fixed", "f32", "f32", 4, 2.0)
    assert entry["binary"] == {"source": "ptx", "ptxas_flags": ["-lineinfo", "-v", f"--gpu-name=sm_{capability()}"]}  # fixed flags, recorded
    assert entry["source_digest_kind"] == "tilefish.binary.v1"
    assert entry["source_digest"] == hashlib.sha256((PTX / "axpy_f32.ptx").read_bytes()).hexdigest()
    assert "ptxas" in entry["toolchain"] and "triton" not in entry["toolchain"]
    x, y = randn((64,), torch.float32, seed=1), randn((64,), torch.float32, seed=2)
    assert same_bytes(tilefish.run(axpy, x, y, alpha=2.0), 2.0 * x + y)
    with pytest.raises(tilefish.TilefishTypeError, match="argument 'alpha' of 'axpy' is fixed to 2.0"):
        tilefish.run(axpy, x, y, alpha=3.0)


def test_replay_of_prebuilt_ptx_is_byte_identical_to_jit(tmp_path):
    m = runtime.export([axpy, strided_copy], tmp_path)
    for size in (5, 1000, 8192):
        x, y = randn((size,), torch.float32, seed=size), randn((size,), torch.float32, seed=size + 7)
        jit = tilefish.run(axpy, x, y)
        replay = tilefish.run(axpy, x, y, mode="replay", manifest=tmp_path, pin=m.digest)
        assert same_bytes(jit, replay)
    wide = randn((33, 200), torch.bfloat16, seed=9)
    view = wide[:, 8:72]  # [33, 64], strides (200, 1): a 64-bit row stride, and a start 16 bytes in
    jit = tilefish.run(strided_copy, view)
    replay = tilefish.run(strided_copy, view, mode="replay", manifest=tmp_path, pin=m.digest)
    assert same_bytes(jit, view.contiguous()) and same_bytes(replay, jit)


def test_a_stride_is_a_64_bit_scalar_in_the_abi(tmp_path):
    (spec,) = compile_specs(strided_copy)
    assert [(a.name, a.kind, a.type) for a in spec.args] == [("x", "pointer", "*bf16"), ("y", "pointer", "*bf16"), ("cols", "fixed", "i32"), ("x_s0", "scalar", "i64")]
    assert spec.args[3].stride_of == (("x", 0),)
    assert spec.binary["kind"] == "ptx" and spec.block == (64, 1, 1) and spec.num_warps == 0
    assert [p.name for p in spec.launch_params] == ["x", "y", "cols", "x_s0"]  # no scratch slots


def test_a_replay_refuses_a_ptx_that_is_not_the_one_the_manifest_was_built_from(tmp_path):
    m = runtime.export(axpy, tmp_path)
    edited = tmp_path / "axpy2.ptx"
    edited.write_bytes((PTX / "axpy_f32.ptx").read_bytes() + b"// the same code, another file\n")

    @tilefish.extern(ptx=edited, symbol="axpy_f32", block=(128,), grid=lambda n: (tilefish.cdiv(n, 128),), name="axpy")
    def axpy_edited(x: Tensor[n, f32], y: Tensor[n, f32], out: Out[Tensor[n, f32]], n_elem: n, alpha: const[f32] = 2.0): ...

    x, y = randn((64,), torch.float32, seed=1), randn((64,), torch.float32, seed=2)
    assert same_bytes(tilefish.run(axpy, x, y, mode="replay", manifest=tmp_path, pin=m.digest), 2.0 * x + y)
    with pytest.raises(TilefishManifestError, match="the file changed since the manifest was written, so its binary is stale"):
        tilefish.run(axpy_edited, x, y, mode="replay", manifest=tmp_path, pin=m.digest)


# -- a cubin is used as it is, for its own architecture ---------------------------------------------------------------------------------------


def make_cubin(directory, ptx_name="axpy_f32.ptx"):
    cap = capability()
    ptx = (PTX / ptx_name).read_bytes()
    import subprocess

    out = directory / f"{ptx_name}.sm_{cap}.cubin"
    subprocess.run([_binary.find_ptxas(), *_binary.PTX_FLAGS, f"--gpu-name=sm_{cap}", str(PTX / ptx_name), "-o", str(out)], check=True, capture_output=True)
    return out, ptx


def test_a_cubin_runs_as_it_is_and_is_refused_for_another_architecture(tmp_path):
    cap = capability()
    cubin, _ = make_cubin(tmp_path)

    @tilefish.extern(cubin={f"sm_{cap}": cubin}, symbol="axpy_f32", block=(128,), grid=lambda n: (tilefish.cdiv(n, 128),), name="axpy_cubin")
    def axpy_cubin(x: Tensor[n, f32], y: Tensor[n, f32], out: Out[Tensor[n, f32]], n_elem: n, alpha: const[f32] = 2.0): ...

    x, y = randn((1000,), torch.float32, seed=3), randn((1000,), torch.float32, seed=4)
    assert same_bytes(tilefish.run(axpy_cubin, x, y), tilefish.run(axpy, x, y))
    m = runtime.export([axpy, axpy_cubin], tmp_path / "m")
    entry = {e.stem: e for e in m.artifacts}["axpy_cubin"].entry
    assert entry["binary"] == {"source": "cubin"} and entry["toolchain"] == {} and entry["source_digest"] == hashlib.sha256(cubin.read_bytes()).hexdigest()
    assert entry["sha256"] == entry["source_digest"]  # the file is the cubin as it was
    assert same_bytes(tilefish.run(axpy_cubin, x, y, mode="replay", manifest=tmp_path / "m", pin=m.digest), tilefish.run(axpy, x, y))
    # the machine code of the cubin ptxas made now and of the one built from the PTX are the same code
    assert {e.stem: e for e in m.artifacts}["axpy"].sha256_code == entry["sha256_code"]
    (spec,) = compile_specs(axpy_cubin)
    other = 80 if cap != 80 else 90
    with pytest.raises(TilefishBuildError, match=rf"has no cubin for sm_{other} \(it has \['sm_{cap}'\]\)"):
        _binary.build_binary(spec, axpy_cubin.binary, arch=other)


def test_a_cubin_for_another_architecture_than_its_key_is_refused(tmp_path):
    cap = capability()
    cubin, _ = make_cubin(tmp_path)
    liar = f"sm_{80 if cap != 80 else 90}"

    @tilefish.extern(cubin={liar: cubin}, symbol="axpy_f32", block=(128,), grid=lambda n: (tilefish.cdiv(n, 128),), name="axpy_liar")
    def axpy_liar(x: Tensor[n, f32], y: Tensor[n, f32], out: Out[Tensor[n, f32]], n_elem: n, alpha: const[f32] = 2.0): ...

    (spec,) = compile_specs(axpy_liar)
    with pytest.raises(TilefishBuildError, match=rf"is for sm_{cap} \(its ELF header says so\)"):
        _binary.build_binary(spec, axpy_liar.binary, arch=int(liar[3:]))


# -- the PTX of a Triton build, declared as a prebuilt extern: the same bytes out -----------------------------------------------------------


def triton_ptx(lowered, directory):
    """The PTX of a lowered kernel from a Triton build (and the build): what Triton hands to ptxas."""
    from tilefish.runtime import load_lowered

    (spec,) = compile_specs(lowered.extern)
    built = build(spec, arch=capability(), fn=load_lowered(lowered), cache=False)
    path = directory / f"{lowered.name}.ptx"
    path.write_text(built.ptx)
    return path, built


def test_the_ptx_of_the_elementwise_gelu_as_a_prebuilt_extern_gives_the_bytes_of_the_triton_path(tmp_path):
    lowered = tilefish.lower(gelu_exact)
    path, built = triton_ptx(lowered, tmp_path)
    assert built.spec.attrs  # the Triton binary was compiled for 16-byte aligned pointers and a multiple of 16 elements

    # Triton's ABI is (x, out, n) and two scratch slots, which a prebuilt extern writes out as explicit, fixed, 64-bit arguments
    @tilefish.extern(ptx=path, symbol=built.symbol, block=(built.num_warps * 32,), shared=built.shared, grid=lambda T: (tilefish.cdiv(T, 1024),), name="gelu_ptx")
    def gelu_ptx(x: Tensor[T, bf16], out: Out[Tensor[T, bf16]], n_: T, global_scratch: const[i64] = 0, profile_scratch: const[i64] = 0): ...

    m = runtime.export([gelu_exact, gelu_ptx], tmp_path / "m")
    entries = {e.stem: e for e in m.artifacts}
    assert entries["gelu_ptx"].entry["abi"]["scratch_slots"] == 0 and entries["gelu_exact"].entry["abi"]["scratch_slots"] == 2
    assert [p["kind"] for p in entries["gelu_ptx"].entry["abi"]["params"]] == ["pointer", "pointer", "scalar", "fixed", "fixed"]
    for size in (16, 1024, 4096, 5 * 1008 + 16 * 3, 100000 - 100000 % 16):
        x = randn((size,), torch.bfloat16, seed=size, scale=3.0)
        triton = tilefish.run(gelu_exact, x)
        jit = tilefish.run(gelu_ptx, x)
        replay = tilefish.run(gelu_ptx, x, mode="replay", manifest=tmp_path / "m", pin=m.digest)
        assert same_bytes(jit, triton), size
        assert same_bytes(replay, triton), size
    # the same PTX, assembled by tilefish's flags, is the machine code Triton's ptxas run made
    assert entries["gelu_ptx"].sha256_code == entries["gelu_exact"].sha256_code


def test_the_ptx_of_the_row_softmax_as_a_prebuilt_extern_gives_the_bytes_of_the_triton_path(tmp_path):
    lowered = tilefish.lower(softmax)
    path, built = triton_ptx(lowered, tmp_path)
    M = Dim("M")

    @tilefish.extern(ptx=path, symbol=built.symbol, block=(built.num_warps * 32,), shared=built.shared, grid=lambda M: (M,), name="softmax_ptx")
    def softmax_ptx(x: Tensor[M, 1000, f32], out: Out[Tensor[M, 1000, f32]], global_scratch: const[i64] = 0, profile_scratch: const[i64] = 0): ...

    m = runtime.export([softmax, softmax_ptx], tmp_path / "m")
    for rows in (1, 9, 130):
        x = randn((rows, 1000), torch.float32, seed=rows, scale=3.0)
        triton = tilefish.run(softmax, x)
        assert same_bytes(tilefish.run(softmax_ptx, x), triton), rows
        assert same_bytes(tilefish.run(softmax_ptx, x, mode="replay", manifest=tmp_path / "m", pin=m.digest), triton), rows


def test_the_scratch_slots_of_a_triton_kernel_are_written_out_or_the_abi_is_refused(tmp_path):
    """A prebuilt extern is launched with exactly its declared parameters: leave the scratch slots out and the PTX entry does not match."""
    lowered = tilefish.lower(gelu_exact)
    path, built = triton_ptx(lowered, tmp_path)

    @tilefish.extern(ptx=path, symbol=built.symbol, block=(built.num_warps * 32,), grid=lambda T: (tilefish.cdiv(T, 1024),), name="gelu_short")
    def gelu_short(x: Tensor[T, bf16], out: Out[Tensor[T, bf16]], n_: T): ...

    (spec,) = compile_specs(gelu_short)
    with pytest.raises(TilefishBuildError, match="does not have the C ABI its stub declares"):
        _binary.build_binary(spec, gelu_short.binary, arch=capability())
