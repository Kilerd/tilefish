"""What Triton reads while it compiles besides the IR, and threads, on a real GPU.

* ``ir_override`` is a compile option that names a file Triton *substitutes* for the IR of a stage: a build with it is not of the kernel's
  IR, so it is refused (as ``TRITON_KERNEL_OVERRIDE`` is). Rewriting that file and building again in a new process used to give the old
  result with no error: the cache key held only the path.
* ``extern_libs`` and ``LLVM_PASS_PLUGIN_PATH`` name files Triton reads (the bitcode libraries it links, a plug-in it loads): the cache key and
  the in-memory builds hold the sha256 of the *content*, so a file that is rewritten under its name is another binary, never a stale hit.
* Triton's frontend and compile are one critical section: builds and digests from several threads give the digests and the machine code of
  the same builds one at a time.
"""

import concurrent.futures
import hashlib
import os
import pathlib

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")
import triton.language as tl  # noqa: E402

import gpu_kernels  # noqa: E402
from gpu_util import randn, same_bytes  # noqa: E402
from lower_cases import gelu_exact, layernorm, softmax, swiglu  # noqa: E402

import tilefish  # noqa: E402
from tilefish import Dim, Out, Tensor, TilefishBuildError, _cache, _ir, f32, runtime  # noqa: E402
from tilefish.aot import build, compile_specs  # noqa: E402

CAP = torch.cuda.get_device_capability()
ARCH = CAP[0] * 10 + CAP[1]
REAL_COMPILE = triton.compile


@pytest.fixture(autouse=True)
def no_builds_in_memory():
    runtime.clear_caches()
    yield
    runtime.clear_caches()


@pytest.fixture
def compiles(monkeypatch):
    calls = []
    monkeypatch.setattr(triton, "compile", lambda *a, **k: (calls.append(1), REAL_COMPILE(*a, **k))[1])
    return calls


def entries():
    return sorted(pathlib.Path(os.environ["TILEFISH_CACHE_DIR"]).rglob("*.json"))


@triton.jit
def scale_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x * 3.0, mask=mask)


N = Dim("N", divisible_by=16)


@tilefish.extern(triton=scale_kernel, num_warps=4, grid=lambda N: (tilefish.cdiv(N, 256),))
def scale(x_ptr: Tensor[N, f32], y_ptr: Out[Tensor[N, f32]], n_elements: N, *, BLOCK: tilefish.const = 256): ...


# -- ir_override -------------------------------------------------------------------------------------------------------------------------


def test_ir_override_with_a_modified_ptx_is_refused_not_run_and_not_cached(compiles, tmp_path):
    """The reviewer's repro: the PTX of ``x * 3.0`` with the constant rewritten to 5.0, handed to Triton as ``ir_override``. Triton would
    compile *that*; the cache key held only the path, so rewriting the file and running in a new process was a stale hit."""
    (spec,) = compile_specs(scale)
    x = torch.ones(64, dtype=torch.float32, device="cuda")
    built = build(spec, arch=ARCH, fn=scale_kernel, cache=False)
    assert "0f40400000" in built.ptx  # 3.0
    replacement = tmp_path / "scale.ptx"
    replacement.write_text(built.ptx.replace("0f40400000", "0f40a00000"))  # 5.0
    count = len(compiles)
    with runtime.build_options(ir_override=str(replacement)):
        with pytest.raises(TilefishBuildError, match=r"the compile option ir_override .*names a file that Triton substitutes for the IR of a stage"):
            tilefish.run(scale, x, mode="jit")
    with pytest.raises(TilefishBuildError, match="ir_override"):
        build(spec, arch=ARCH, fn=scale_kernel, options={"ir_override": str(replacement)})
    assert len(compiles) == count and entries() == []  # nothing was compiled, nothing cached
    replacement.write_text(built.ptx.replace("0f40400000", "0f40e00000"))  # 7.0: a rewrite of the file, then and now the same refusal
    with runtime.build_options(ir_override=str(replacement)):
        with pytest.raises(TilefishBuildError, match="ir_override"):
            tilefish.run(scale, x, mode="jit")
    assert torch.equal(tilefish.run(scale, x, mode="jit"), x * 3)  # and the kernel is what it says


# -- extern_libs ------------------------------------------------------------------------------------------------------------------------


def default_libdevice():
    path = pathlib.Path(triton.__file__).parent / "backends" / "nvidia" / "lib" / "libdevice.10.bc"
    if not path.exists():
        pytest.skip("this Triton has no libdevice at its default place")
    return path


def test_a_rewritten_extern_library_is_another_binary_in_memory_and_on_disk(compiles, tmp_path):
    """``libdevice`` handed as ``extern_libs`` and rewritten under the same path: never the binary of the other content."""
    original = default_libdevice().read_bytes()
    library = tmp_path / "libdevice.10.bc"
    library.write_bytes(original)
    x = randn((1 << 12,), torch.bfloat16, 0, scale=3.0)
    expected = tilefish.run(gpu_kernels.gelu_erf, x, mode="jit")  # the default libdevice
    base = len(compiles)
    options = {"extern_libs": {"libdevice": str(library)}}
    with runtime.build_options(**options):
        assert same_bytes(tilefish.run(gpu_kernels.gelu_erf, x, mode="jit"), expected)  # the same content: the same results
        built = len(compiles)
        assert built == base + 1  # (another key: the option is in it)
        tilefish.run(gpu_kernels.gelu_erf, x, mode="jit")
        assert len(compiles) == built  # the memo
        library.write_bytes(b"this is not bitcode")  # the same path, another file
        with pytest.raises(TilefishBuildError, match="Triton could not compile"):  # a stale hit would have run
            tilefish.run(gpu_kernels.gelu_erf, x, mode="jit")
        assert len(compiles) == built + 1
        library.write_bytes(original)  # put back: served again, by memory and by disk, with no compile
        runtime.clear_caches()
        assert same_bytes(tilefish.run(gpu_kernels.gelu_erf, x, mode="jit"), expected)
        assert len(compiles) == built + 1
    (spec,) = compile_specs(gpu_kernels.gelu_erf)
    key_options = {**spec.ast_source_inputs().options, **options}
    assert _cache.compile_files(key_options) == {"extern_libs.libdevice": hashlib.sha256(original).hexdigest()}
    assert len(entries()) == 2  # the default libdevice, and this one: nothing was stored for the file that was not bitcode


def test_llvm_pass_plugin_path_is_in_the_key_by_the_content_of_the_file(monkeypatch, tmp_path):
    """No plug-in is loaded here (a junk file would not load): what is checked is the key, which is where a stale hit would come from."""
    plugin = tmp_path / "plugin.so"
    plugin.write_bytes(b"one")
    base = _cache.codegen_environment()
    monkeypatch.setenv("LLVM_PASS_PLUGIN_PATH", str(plugin))
    first = _cache.codegen_environment()
    assert first["env_files"] == {"LLVM_PASS_PLUGIN_PATH": hashlib.sha256(b"one").hexdigest()} and "env_files" not in base
    plugin.write_bytes(b"two")  # rewritten under the same path
    second = _cache.codegen_environment()
    assert second["env_files"] != first["env_files"] and second["env"] == first["env"]
    plugin.unlink()
    with pytest.raises(TilefishBuildError, match=r"LLVM_PASS_PLUGIN_PATH .*cannot be read"):
        _cache.codegen_environment()


def test_the_knobs_and_files_the_audit_found_are_what_the_key_reads_on_this_triton():
    """The names are Triton 3.5.1's; a Triton that renames one makes the key say so (``<AttributeError>``) instead of dropping it."""
    environment = _cache.codegen_environment()
    for name in ("compilation.override", "compilation.use_ir_loc", "language.default_fp_fusion", "nvidia.libdevice_path", "nvidia.mock_ptx_version", "runtime.override_arch"):
        assert name in environment["knobs"] and not environment["knobs"][name].startswith("<"), name


# -- threads -----------------------------------------------------------------------------------------------------------------------------


def test_threads_that_build_and_digest_at_once_get_what_one_thread_gets(compiles):
    """The frontend and the compile are one critical section, so several threads do not run Triton's frontend at the same time (a compile's
    own run of it included) and every digest and every cubin is the one a single thread makes."""
    kernels = [softmax, layernorm, gelu_exact, swiglu]
    jobs = []
    for k in kernels:
        low = k.lower()
        (spec,) = low.specs()
        jobs.append((spec, runtime.load_lowered(low)))
    single = [(runtime._ir_digest_of(spec, fn, ARCH, {}), build(spec, arch=ARCH, fn=fn, cache=False).sha256_code) for spec, fn in jobs]
    count = len(compiles)

    def work(i):
        spec, fn = jobs[i % len(jobs)]
        digest = runtime._ir_digest_of(spec, fn, ARCH, {})
        built = build(spec, arch=ARCH, fn=fn, cache=False)
        again = runtime._ir_digest_of(spec, fn, ARCH, {})
        return i % len(jobs), digest, built.ir_digest, built.sha256_code, again

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(32)))
    for index, digest, built_digest, code, again in results:
        assert digest == built_digest == again == single[index][0]
        assert code == single[index][1]
    assert len(compiles) == count + 32  # every build compiled (cache=False), none failed the check that its IR is the one it was identified by
    assert _ir.LOCK is tilefish._build._compile_lock
