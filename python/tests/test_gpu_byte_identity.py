"""The central claim of tilefish's Python side, on a real GPU: ``replay`` runs the exact machine code that a C++
or Rust host would load, and it agrees byte for byte with ``jit``.

``jit`` builds the current source through its compile spec (``aot.build``, in memory) and launches the binary
through the CUDA driver; ``replay`` loads the prebuilt cubins of a manifest directory (``runtime.export``), checks
them against the digests of the manifest and launches them through the same driver path. Both get the same inputs;
the outputs *and* the tensors that are updated in place are compared as raw bytes (``t.view(torch.uint8)``,
``torch.equal``), over several shapes and seeds, for

* the five goldens of the lowering (softmax, layernorm, the README's residual add + rmsnorm with its
  ``InOut`` tensor, the exact gelu, swiglu),
* a row kernel with a bucketed row length, over each of its buckets (one binary per bucket, no padding),
* two hand-written Triton kernels declared with ``extern`` (an rmsnorm and an exact-erf gelu), and an
  in-place ``y += alpha * x``.

The other half of the gate is the machine code: a cubin that ``jit`` builds has the ``sha256_code`` of the one in
the manifest, also when the manifest was written by another process from another directory
(``test_gpu_build.py``). Triton's native JIT launcher is compared too, as a test-only helper (``native_jit.py``).
"""

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import gpu_kernels  # noqa: E402
from conftest import REPO_ROOT, make_my_kernels, readme_snippet  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402
from native_jit import native_jit  # noqa: E402
from lower_cases import M, T, gelu_exact, layernorm, softmax, swiglu  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Out, Tensor, TilefishPreconditionError, TilefishTypeError, bf16, const, f32  # noqa: E402
from tilefish import runtime  # noqa: E402

BF16, F32 = torch.bfloat16, torch.float32
SEEDS = (0, 1, 2)

# -- the kernels -------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def readme_kernel():
    if not (REPO_ROOT / "README.md").exists():
        pytest.skip("README.md is not available (not running from a source checkout)")
    ns = {"my_kernels": make_my_kernels()}
    exec(compile(readme_snippet(), "README.md", "exec"), ns)
    return ns["residual_add_rmsnorm"]


Db = Dim("Db", buckets=[96, 512, 1000])


@tilefish.kernel
def bucketed_rms(x: Tensor[M, Db, bf16], w: Tensor[Db, bf16], *, eps: const[f32] = 1e-5) -> Tensor[M, Db, bf16]:
    xf = x.to(f32)
    mean_square = tilefish.mean(xf * xf, axis=-1, keepdims=True)
    return (xf * tilefish.rsqrt(mean_square + eps) * w.to(f32)).to(bf16)


@tilefish.kernel
def out_and_return(x: Tensor[T, f32], y: Out[Tensor[T, f32]]) -> Tensor[T, f32]:
    y[...] = x * 2.0 + 1.0
    return x - 0.5


# a kernel parameter called `grid` collides with Triton's launch options: the lowering renames it in the generated function
@tilefish.kernel
def named_like_a_launch_option(grid: Tensor[T, f32], num_warps: const[f32] = 0.5) -> Tensor[T, f32]:
    return grid * num_warps


def args_softmax(rows, seed):
    return (randn((rows, 1000), F32, seed, scale=3.0),)


def args_layernorm(rows, seed):
    return (randn((rows, 768), BF16, seed), randn((768,), BF16, seed + 100), randn((768,), BF16, seed + 200, scale=0.1))


def args_rmsnorm_golden(rows, seed):
    return (randn((rows, 2560), BF16, seed), randn((rows, 2560), F32, seed + 50), randn((2560,), BF16, seed + 100))


def args_gelu(n, seed):
    return (randn((n,), BF16, seed, scale=3.0),)


def args_swiglu(rows, seed):
    return (randn((rows, 3072), BF16, seed, scale=2.0), randn((rows, 3072), BF16, seed + 100))


def args_bucketed(d):
    return lambda rows, seed: (randn((rows, d), BF16, seed), randn((d,), BF16, seed + 100))


def args_rmsnorm_extern(rows, seed):
    return (randn((rows, 2048), BF16, seed), randn((2048,), BF16, seed + 100))


def args_gelu_erf(n, seed):
    return (randn((n,), BF16, seed, scale=3.0),)


def args_axpy(n, seed):
    return (randn((n,), F32, seed), randn((n,), F32, seed + 100))


def args_flat(n, seed):
    return (randn((n,), F32, seed),)


ROW_SIZES = (1, 7, 64, 129)
FLAT_SIZES = (16, 1040, 65552, 1 << 20)


def cases():
    yield "softmax", lambda: softmax, args_softmax, ROW_SIZES
    yield "layernorm", lambda: layernorm, args_layernorm, ROW_SIZES
    yield "residual_add_rmsnorm", "readme", args_rmsnorm_golden, (1, 3, 33)
    yield "gelu_exact", lambda: gelu_exact, args_gelu, FLAT_SIZES
    yield "swiglu", lambda: swiglu, args_swiglu, (1, 3, 33, 130)
    for d in (96, 512, 1000):
        yield f"bucketed_rms.D{d}", lambda: bucketed_rms, args_bucketed(d), (1, 5, 40)
    yield "extern rmsnorm", lambda: gpu_kernels.rmsnorm, args_rmsnorm_extern, (1, 5, 64)
    yield "extern gelu_erf", lambda: gpu_kernels.gelu_erf, args_gelu_erf, FLAT_SIZES
    yield "extern axpy (InOut)", lambda: gpu_kernels.axpy, args_axpy, FLAT_SIZES
    yield "out_and_return", lambda: out_and_return, args_flat, FLAT_SIZES
    yield "renamed launch-option parameter", lambda: named_like_a_launch_option, args_flat, FLAT_SIZES


def all_runs():
    for name, op, make, sizes in cases():
        for size in sizes:
            for seed in SEEDS:
                yield pytest.param(name, op, make, size, seed, id=f"{name}-{size}-seed{seed}")


def tensors_of(result):
    if result is None:
        return []
    return list(result) if isinstance(result, tuple) else [result]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """``exported(op)``: the manifest directory with every binary of ``op``, built once per operator."""
    cache = {}

    def get(op):
        if id(op) not in cache:
            cache[id(op)] = runtime.export(op, tmp_path_factory.mktemp("manifest")).parent
        return cache[id(op)]

    return get


def run_in(mode, op, args, runner=None, **kw):
    """Run on clones of the arguments (so that every mode starts from the same bytes); return the results and the clones."""
    clones = [a.clone() if torch.is_tensor(a) else a for a in args]
    return (runner or (lambda *a, **k: tilefish.run(op, *a, mode=mode, **k)))(*clones, **kw), clones


@pytest.mark.parametrize("name, op, make, size, seed", list(all_runs()))
def test_jit_and_replay_agree_byte_for_byte(name, op, make, size, seed, request, exported):
    op = request.getfixturevalue("readme_kernel") if op == "readme" else op()
    args = make(size, seed)
    jit, jit_args = run_in("jit", op, args)
    replay, replay_args = run_in("replay", op, args, manifest=exported(op))

    outs_jit, outs_replay = tensors_of(jit), tensors_of(replay)
    assert len(outs_jit) == len(outs_replay)
    for a, b in zip(outs_jit, outs_replay):
        assert same_bytes(a, b), f"{name}: an output differs between jit and replay"
        assert torch.isfinite(a.float()).all() and a.float().abs().sum() > 0  # something was computed
    # the arguments after the call: the in-place ones are updated, and identically; the read-only ones are untouched
    inout = {p.name for p in op.signature.params if p.kind == "tensor" and p.access == "inout"}
    names = [p.name for p in op.signature.params if p.kind == "tensor" and p.access != "out"]
    for pname, original, a, b in zip(names, args, jit_args, replay_args):
        assert same_bytes(a, b), f"{name}: argument '{pname}' differs between jit and replay"
        if pname not in inout:
            assert same_bytes(a, original), f"{name}: the read-only argument '{pname}' was modified"
    for pname, original, a in zip(names, args, jit_args):
        if pname in inout:
            assert not same_bytes(a, original), f"{name}: the in-place argument '{pname}' was not updated"


def test_jit_builds_in_memory_and_replay_loads_the_manifests_cubin(tmp_path):
    from tilefish import _cuda

    args = args_softmax(7, 0)
    runtime.clear_caches()
    assert not runtime._builds
    tilefish.run(softmax, *args, mode="jit")
    (built,) = runtime._builds.values()  # jit built the current source through its spec, in memory ...
    assert built.cubin[:4] == b"\x7fELF" and built.symbol == "tf_softmax" and built.ptx.startswith("//")
    assert [k.built.sha256 for k in _cuda._loaded.values()] == [built.sha256]  # ... and launched that binary
    assert not list(tmp_path.iterdir())  # nothing is written

    directory = runtime.export(softmax, tmp_path).parent
    runtime.clear_caches()
    tilefish.run(softmax, *args, mode="replay", manifest=directory)
    assert not runtime._builds  # replay builds nothing
    (loaded,) = _cuda._loaded.values()
    assert loaded.built.cubin == (directory / built.cubin_name).read_bytes()
    assert loaded.built.ptx == ""  # the manifest carries the cubin, not the PTX
    assert loaded.built.sha256_code == built.sha256_code  # the same machine code as the in-memory build


def test_triton_native_jit_computes_the_same_bytes_too(exported):
    """Triton's own launcher specializes on argument values, so this is a comparison, not a mode."""
    for name, op, make, sizes in cases():
        if op == "readme" or name.startswith(("out_and", "renamed")):
            continue
        fn = op()
        args = make(sizes[1], 0)
        native, native_args = run_in("jit", fn, args, runner=lambda *a, fn=fn: native_jit(fn, *a))
        replay, replay_args = run_in("replay", fn, args, manifest=exported(fn))
        for a, b in zip(tensors_of(native) + native_args, tensors_of(replay) + replay_args):
            assert same_bytes(a, b), name


@pytest.fixture
def go(exported):
    """``go(mode, op, *args, **kw)``: ``tilefish.run`` in a mode, with the manifest of the operator for ``replay``."""

    def call(mode, op, *args, **kw):
        if mode == "replay":
            kw["manifest"] = exported(op)
        return tilefish.run(op, *args, mode=mode, **kw)

    return call


def test_op_run_and_lowered_run_are_the_same_as_tilefish_run(exported):
    args = args_layernorm(5, 3)
    manifest = exported(layernorm)
    a = tilefish.run(layernorm, *args, mode="replay", manifest=manifest)
    assert same_bytes(a, layernorm.run(*args, mode="replay", manifest=manifest))
    assert same_bytes(a, layernorm.lower().run(*args, mode="replay", manifest=manifest))
    assert same_bytes(a, tilefish.run(layernorm.lower(), *args, mode="jit"))
    assert same_bytes(a, layernorm.run(*args))  # jit is the default mode


def test_results_are_returned_as_in_a_trace(go):
    x = randn((64,), F32, 0)
    for mode in ("jit", "replay"):
        returned, out = go(mode, out_and_return, x)  # the returned value, then the Out tensor
        assert same_bytes(returned, x - 0.5) and same_bytes(out, x * 2.0 + 1.0)
        only = go(mode, gpu_kernels.gelu_erf, randn((16,), BF16, 0))  # an extern: its Out tensors
        assert only.shape == (16,) and only.dtype == BF16


@pytest.mark.parametrize("mode", ["jit", "replay"])
def test_the_cuda_stream_is_torchs_current_stream(mode, go):
    # both modes launch on torch's current stream, so they order with torch's own kernels without a synchronize
    x = randn((1 << 20,), BF16, 1, scale=3.0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())  # x exists before the side stream reads it
    with torch.cuda.stream(stream):
        y = x * 1.0
        out = go(mode, gpu_kernels.gelu_erf, y)
        z = out.float() + 1.0
    stream.synchronize()
    expected = tilefish.run(gpu_kernels.gelu_erf, x, mode="jit").float() + 1.0
    assert same_bytes(z, expected)


def test_a_manifest_is_required_by_replay_and_only_by_replay(exported):
    from tilefish import TilefishRuntimeError

    x = randn((16,), BF16, 0)
    with pytest.raises(TilefishRuntimeError, match="replay runs prebuilt cubins from a manifest directory"):
        tilefish.run(gpu_kernels.gelu_erf, x, mode="replay")
    with pytest.raises(TilefishRuntimeError, match="manifest= is for mode='replay'"):
        tilefish.run(gpu_kernels.gelu_erf, x, mode="jit", manifest=exported(gpu_kernels.gelu_erf))
    with pytest.raises(TilefishRuntimeError, match="manifest= is for mode='replay'"):
        tilefish.run(layernorm, *args_layernorm(2, 0), mode="reference", manifest=exported(layernorm))


# -- preconditions, on real tensors ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["jit", "replay"])
def test_a_misaligned_pointer_is_refused_before_launch(mode, go):
    base = randn((64 + 8,), BF16, 0)
    misaligned = base[4 : 4 + 64]  # 8 bytes past a 16-byte boundary
    assert misaligned.data_ptr() % 16 == 8
    with pytest.raises(TilefishPreconditionError, match="not aligned to 16 bytes") as e:
        go(mode, gpu_kernels.gelu_erf, misaligned)
    assert e.value.kind == "pointer_alignment"


@pytest.mark.parametrize("mode", ["jit", "replay"])
def test_a_size_that_is_not_a_bucket_is_refused_there_is_no_padding(mode, go):
    x, w = args_bucketed(96)(3, 0)
    go(mode, bucketed_rms, x, w)
    with pytest.raises(TilefishPreconditionError, match="no padding") as e:
        go(mode, bucketed_rms, randn((3, 100), BF16, 0), randn((100,), BF16, 1))
    assert e.value.kind == "bucket"
    assert "[96, 512, 1000]" in str(e.value)


@pytest.mark.parametrize("mode", ["jit", "replay"])
def test_other_preconditions_are_refused_too(mode, go):
    x = randn((60,), BF16, 0)
    with pytest.raises(TilefishPreconditionError, match="divisible_by=16") as e:
        go(mode, gpu_kernels.gelu_erf, x)
    assert e.value.kind == "divisible"
    with pytest.raises(TilefishPreconditionError, match="dtype") as e:
        go(mode, gpu_kernels.gelu_erf, randn((64,), F32, 0))
    assert e.value.kind == "dtype"
    with pytest.raises(TilefishPreconditionError, match="not contiguous") as e:
        go(mode, gpu_kernels.gelu_erf, randn((64, 2), BF16, 0)[:, 0])
    assert e.value.kind == "contiguous"
    with pytest.raises(TilefishPreconditionError, match="requires 2048") as e:
        go(mode, gpu_kernels.rmsnorm, randn((3, 2000), BF16, 0), randn((2000,), BF16, 1))
    assert e.value.kind == "static_size"
    with pytest.raises(TilefishPreconditionError, match="overlaps") as e:
        a = randn((64,), F32, 0)
        go(mode, gpu_kernels.axpy, a, a)
    assert e.value.kind == "alias"
    with pytest.raises(TilefishTypeError, match="cannot be changed"):
        go(mode, gpu_kernels.rmsnorm, *args_rmsnorm_extern(3, 0), EPS=1e-3)


def test_cpu_tensors_do_not_run_on_a_binary(go):
    from tilefish import TilefishRuntimeError

    for mode in ("jit", "replay"):
        with pytest.raises(TilefishRuntimeError, match="CUDA device"):
            go(mode, gpu_kernels.gelu_erf, torch.zeros(16, dtype=BF16))
    with pytest.raises(TilefishPreconditionError, match="different devices"):
        go("jit", layernorm, torch.zeros(2, 768, dtype=BF16), randn((768,), BF16, 0), randn((768,), BF16, 1))
