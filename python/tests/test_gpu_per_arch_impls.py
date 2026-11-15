"""Implementations per architecture with names, prebuilt binaries and ABIs of their own, on a real GPU (the ops of ``per_arch_impl_kernels.py``).

``copy2d``, ``copy2d_ptx`` and ``copy2d_rows`` are one public signature, ``y = x`` for a view of f16 rows, each with a TMA descriptor form (Triton, its
own stub, named ``tma``), a pointer form (Triton, the op's stub) or a PTX form (a prebuilt binary, its own stub, named ``ptx``), on other
architectures. Held here, on the device's architecture:

* ``jit`` and ``replay`` run the implementation of the device, with its own ABI, and give the input's bytes, for a dense tensor and a view;
* each manifest entry has that implementation's key (a name, the architecture or ``"fallback"``), its kind and its ABI (descriptors, or pointers and
  scalars), and a named implementation built for sm_100 and sm_103 is one declaration for both;
* a precondition one implementation has (a TMA descriptor's 16-byte rows) refuses a call only where that implementation runs;
* a plan runs the implementation of the device in a step that reads a view of an arena buffer, in jit, in replay and in a capture, and a plan is refused
  when any implementation could not run a step, whichever runs here;
* a cubin implementation (made here by ptxas for the device's architecture) runs, and one sent to an architecture it has no cubin for is refused.

The tests that need an sm_120 GPU are marked so (``test_on_sm_120_...``): the fallbacks run there.
"""

import subprocess

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:  # pragma: no cover - TMA needs sm_90
    pytest.skip("TMA descriptors need sm_90 or newer", allow_module_level=True)

import per_arch_impl_kernels as pk  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Out, StrideOf, Strided, Tensor, TilefishDeclarationError, TilefishPreconditionError, const, f16, runtime  # noqa: E402
from tilefish._binary import find_ptxas  # noqa: E402
from tilefish.aot import compile_specs, declaration_digest  # noqa: E402

F16 = torch.float16
CAP = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
CC = CAP[0] * 10 + CAP[1]
ARCH = f"sm_{CC}"
# what each op runs on the architectures the tests know: (its key, kind, how it takes its tensors)
EXPECTED = {
    103: {"copy2d": ("tma", "triton", "tensordesc"), "copy2d_ptx": ("ptx", "binary", "pointer"), "copy2d_rows": ("sm_103", "triton", "pointer")},
    120: {"copy2d": ("fallback", "triton", "pointer"), "copy2d_ptx": ("fallback", "triton", "pointer"), "copy2d_rows": ("fallback", "triton", "tensordesc")},
}


def inputs():
    """A dense ``[100, 192]`` tensor and the right half of a ``[100, 384]`` one (a view: rows of 768 bytes, 384 bytes in)."""
    wide = randn((100, 384), F16, seed=1)
    return {"dense": randn((100, 192), F16, seed=2), "view": wide.split(192, dim=-1)[1]}


def takes_descriptors(op):
    (spec,) = compile_specs(op, arch=CC)
    return any(a.kind == "tensordesc" for a in spec.args)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("per_arch_impls")
    return directory, runtime.export(list(pk.OPS.values()), directory)


@pytest.mark.parametrize("which", ["dense", "view"])
@pytest.mark.parametrize("name", list(pk.OPS))
def test_jit_and_replay_run_the_implementation_of_the_device_with_its_abi(exported, name, which):
    directory, manifest = exported
    op, x = pk.OPS[name], inputs()[which]
    jit = tilefish.run(op, x)
    replay = tilefish.run(op, x, mode="replay", manifest=directory, pin=manifest.digest)
    assert same_bytes(jit, x.contiguous()) and same_bytes(replay, jit)
    assert same_bytes(tilefish.run(op, x, mode="reference"), x.contiguous())


def test_each_entry_has_the_key_the_kind_and_the_abi_of_its_implementation(exported, capsys):
    _, manifest = exported
    found = {}
    for name, op in pk.OPS.items():
        one = op.for_arch(CC)
        entry = manifest.get(name, ARCH).entry
        assert entry["implementation"] == one.implementation and entry["declaration_digest"] == declaration_digest(one)
        kinds = [p["kind"] for p in entry["abi"]["params"]]
        if entry["kind"] == "binary":
            assert kinds == ["pointer", "pointer", "scalar", "scalar"] and entry["abi"]["params"][3]["type"] == "i64"
            assert entry["abi"]["scratch_slots"] == 0 and entry["binary"]["source"] == "ptx"
        elif takes_descriptors(op):
            assert kinds == ["tensordesc", "tensordesc"] and entry["constexprs"] == [{"index": 2, "name": "BM", "value": 64}, {"index": 3, "name": "BN", "value": 64}]
        else:
            assert kinds == ["pointer", "pointer", "scalar", "scalar"]
        found[name] = (entry["implementation"], entry["kind"], kinds[0])
    with capsys.disabled():
        print(f"\n{ARCH}: {found}")
    if CC in EXPECTED:
        assert found == EXPECTED[CC]


def test_a_named_implementation_is_one_declaration_for_every_architecture_mapped_to_it(tmp_path):
    """Built for sm_100 and sm_103 (the device need not be either), the TMA form and the PTX form each have one declaration digest and one spec."""
    for arch in (100, 103):
        manifest = runtime.export([pk.copy2d, pk.copy2d_ptx], tmp_path, arch=arch)
    for name, key in (("copy2d", "tma"), ("copy2d_ptx", "ptx")):
        a, b = manifest.get(name, "sm_100").entry, manifest.get(name, "sm_103").entry
        assert a["implementation"] == b["implementation"] == key and a["declaration_digest"] == b["declaration_digest"]
        assert compile_specs(pk.OPS[name], arch=100) == compile_specs(pk.OPS[name], arch=103)


def test_a_precondition_of_one_implementation_refuses_a_call_only_where_it_runs():
    """Rows of 130 f16 (260 bytes) are no multiple of 16 bytes: a TMA descriptor cannot be made of them, a pointer takes them."""
    x = randn((70, 130), F16, seed=3)
    for name, op in pk.OPS.items():
        if takes_descriptors(op):
            with pytest.raises(TilefishPreconditionError) as e:
                tilefish.run(op, x)
            assert e.value.kind == "tensor_map", name
        else:
            assert same_bytes(tilefish.run(op, x), x), name


@pytest.mark.parametrize("name", list(pk.OPS))
def test_a_plan_runs_the_implementation_of_the_device_in_a_step_of_a_view(tmp_path, name):
    g = pk.doubled_copy(pk.OPS[name])
    p = runtime.export(g, tmp_path / "plan")
    entry = tilefish.Manifest.load(tmp_path / "plan").get(name, ARCH).entry
    assert entry["implementation"] == pk.OPS[name].for_arch(CC).implementation
    kw = {"mode": "replay", "plan": tmp_path / "plan", "pin": p.digest}
    for rb in (64, 192):
        x = randn((rb, 384), F16, seed=rb)
        want = (x * 2)[:, 192:].contiguous()
        jit = tilefish.run(g, x)
        replay = tilefish.run(g, x, **kw)
        assert same_bytes(jit, want) and same_bytes(replay, want)
        out = torch.empty_like(want)
        cap = tilefish.capture(g, x, outputs=out, **kw)
        cap.replay()
        torch.cuda.synchronize()
        assert same_bytes(out, want)


def test_a_plan_is_refused_when_any_implementation_could_not_run_a_step():
    """The right half of a ``[Rb, 196]`` tensor starts 196 bytes in and has rows of 392 bytes: the pointer form takes it, a TMA descriptor cannot."""
    R2 = Dim("R2", buckets=[64])
    for op in (pk.copy2d, pk.copy2d_rows):  # (each has a TMA form on some architecture and a pointer form on another)

        @tilefish.graph(name=f"odd_{op.name}")
        def odd(x: Tensor[R2, 196, f16]) -> Tensor[R2, 98, f16]:
            return op(x.split(98, axis=-1)[1])

        with pytest.raises(TilefishPreconditionError, match=r"its implementation '(tma|fallback)' \(a plan is for every architecture\)"):
            tilefish.plan(odd)


def test_a_cubin_implementation_for_this_architecture(tmp_path):
    cubin = tmp_path / f"copy.{ARCH}.cubin"
    done = subprocess.run([find_ptxas(), f"--gpu-name={ARCH}", str(pk.PTX), "-o", str(cubin)], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr
    impl = tilefish.impl(cubin={ARCH: cubin}, symbol="strided_copy_bf16", block=(64,), stub=pk.copy_ptx_stub)
    R, C = pk.R, pk.C

    @tilefish.extern(by_arch={ARCH: impl}, fallback=pk.ROWS, grid=lambda R: (R,), name="copy2d_cubin")
    def copy2d_cubin(x: Strided[Tensor[R, C, f16]], y: Out[Tensor[R, C, f16]], n_cols: C, s_row: StrideOf("x", 0), BLOCK: const = 256): ...

    x = inputs()["view"]
    jit = tilefish.run(copy2d_cubin, x)
    manifest = runtime.export(copy2d_cubin, tmp_path / "m")
    entry = manifest.get("copy2d_cubin", ARCH).entry
    assert (entry["kind"], entry["implementation"], entry["binary"]) == ("binary", ARCH, {"source": "cubin"})
    assert entry["source_digest"] == entry["sha256"]
    replay = tilefish.run(copy2d_cubin, x, mode="replay", manifest=tmp_path / "m", pin=manifest.digest)
    assert same_bytes(jit, x.contiguous()) and same_bytes(replay, jit)
    other = "sm_120" if CC != 120 else "sm_103"
    with pytest.raises(TilefishDeclarationError, match=f"is a cubin for {ARCH}, but by_arch= sends {other} to it"):

        @tilefish.extern(by_arch={ARCH: impl, other: impl}, grid=lambda R: (R,), name="copy2d_cubin")
        def nowhere(x: Strided[Tensor[R, C, f16]], y: Out[Tensor[R, C, f16]], n_cols: C, s_row: StrideOf("x", 0), BLOCK: const = 256): ...


@pytest.mark.skipif(CC != 120, reason="needs an sm_120 GPU: there the fallbacks run (the pointer form of copy2d and copy2d_ptx, the TMA form of copy2d_rows)")
def test_on_sm_120_the_fallbacks_run(exported):
    _, manifest = exported
    assert {n: manifest.get(n, "sm_120").entry["implementation"] for n in pk.OPS} == {n: "fallback" for n in pk.OPS}
    x = inputs()["view"]
    for op in pk.OPS.values():
        assert same_bytes(tilefish.run(op, x), x.contiguous())
