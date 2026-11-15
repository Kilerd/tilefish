"""The operators the Rust host is tested with (not collected: the name has no ``test_`` prefix).

``ops()``: five lowered kernels (the goldens of the lowering: softmax, layernorm, the README's residual add + rmsnorm with its
``InOut`` tensor, the exact gelu, swiglu), a row kernel with a bucketed row length (one binary per bucket), three hand-written Triton
kernels declared with ``tilefish.extern`` (an rmsnorm, an exact-erf gelu and an in-place ``y += alpha * x``: ``gpu_kernels.py``; and
``mm_kernels.py`` for the matrix product), then what manifest schema 3 added (``gpu_kernels_p1.py``):

* strided views: ``head_copy`` (a view that starts at any element), ``head_copy16`` (compiled for a 16-byte aligned pointer),
  ``head_copy64`` (64-bit strides) and ``softmax3d`` (a row softmax over a permuted view);
* device scalars: ``masked_rows`` (a hand-written kernel), ``scale_by`` and ``scale_flat`` (lowered kernels that ``tilefish.load`` them);
* prebuilt binaries that need no toolchain to be declared: the hand-written PTX of ``ptx/`` (``axpy_ptx``: a float fixed argument;
  ``strided_copy``: a 64-bit stride and a fixed ``i32``);

and what M3.5 added (``types_kernels.py``): ``add_one_wide``, a Triton kernel whose count is an ``i64[n]`` scalar; ``scale_null`` and ``scale_bias``, one
Triton kernel declared with a null pointer (``Null[f32]``) and with a tensor; ``scale_null_ptx`` and ``scale_bias_ptx``, the same of a hand-written PTX
that takes a 64-bit count; ``add_rows_ua``, the specialisation of a lowered kernel for a first pointer aligned to its element only
(``KernelOp.unaligned``); and what M3.3 added (``gpu_kernels_p4.py``): ``fma_plain`` and ``fma_fused``, one hand-written kernel declared with ``enable_fp_fusion=False``
and ``True``; ``qkv_sum`` and ``qkv_sum16``, a kernel that reads three views with one stride pair (``StrideOf(("q", "k", "v"), axis)``);
``key_mask`` and ``key_mask_any``, a ``[B]`` mask made from a device scalar, whose ``B`` only the ``Out`` tensor determines; and
(``gpu_kernels_p5.py``) ``add_into``, an in-place kernel with two pointers whose ``Out`` may be written over an input (``MayAlias``).

``gpu_ops(directory, arch)`` is what needs the GPU toolchain to *declare*: the PTX of two lowered goldens taken from a Triton build
(``gelu_ptx``, ``softmax_ptx``: the same machine code as the Triton path, declared as prebuilt externs with the Triton ABI's two scratch
slots written out as fixed arguments) and a cubin (``axpy_cubin``) that ptxas makes from the hand-written PTX. A declaration reads its
file once, so each is written to ``directory`` first (for ``arch``: the PTX of a Triton build is for one architecture, and a cubin
is used for its own); the files persist there, next to the manifest the caller exports. ``library_ops()`` is the pinned cuBLASLt GEMMs
(``gemm`` and ``gemm_bias`` over the buckets and shapes of ``test_gpu_gemm.py``).

``ops()`` and ``library_ops()`` work without Triton, torch or a GPU: ``gpu_kernels`` only needs ``triton.jit`` and ``tl.constexpr`` to be
*declared*, and a declaration (what ``emit`` types a launcher from) does not run a kernel, so where Triton is not installed a stand-in module
is there while it is imported. Its declarations are the real ones: the Triton function is named by its module and name, and that is all a
declaration records of it.

``gluon_ops()`` is the Gluon externs of ``gluon_kernels.py`` (``row_dot`` and ``row_dot_fma``: a layout-pinned row reduction over two views read with
one ``i32`` stride, declared with ``enable_fp_fusion=False`` and ``True``); importing them needs Triton's Gluon, so only the gate and the gpu tests
call it, and the committed fixtures do not have them.

``generated_ops()`` is four ops whose Triton tilefish generates (``tilefish.im2col1d`` in its two column orders, and ``tilefish.col2im1d`` in its
two column orders and layouts): strided externs like any other to the Rust host. Only a fresh export of the gate has them.

``export_rust_fixtures.py`` builds these into the fixtures of the crate, and ``rust_gate.py`` runs them in Python (``replay``) and in Rust.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import pathlib
import sys
import types

from conftest import make_my_kernels, readme_snippet
from lower_cases import M, T, gelu_exact, layernorm, softmax, swiglu

import tilefish
from tilefish import Dim, Out, StrideOf, Strided, Tensor, bf16, const, f32, i32, i64, runtime

Db = Dim("Db", buckets=[96, 512, 1000])

PTX = pathlib.Path(__file__).resolve().parent / "ptx"
GEMM_BUCKETS = [32, 64, 320]
GEMM_SHAPES = [(1024, 1024), (3072, 1024), (1024, 3072), (4096, 3072), (3072, 4096)]  # (N, K): the shapes of test_gpu_gemm.py
GEMM_SHAPES_COMMITTED = GEMM_SHAPES[:2]  # the fixtures of the crate hold these two (the gate exports all of them)


@tilefish.kernel
def bucketed_rms(x: Tensor[M, Db, bf16], w: Tensor[Db, bf16], *, eps: const[f32] = 1e-5) -> Tensor[M, Db, bf16]:
    xf = x.to(f32)
    mean_square = tilefish.mean(xf * xf, axis=-1, keepdims=True)
    return (xf * tilefish.rsqrt(mean_square + eps) * w.to(f32)).to(bf16)


def readme_kernel():
    """``residual_add_rmsnorm`` of the README (an ``InOut`` tensor), run from the README's own code block."""
    ns = {"my_kernels": make_my_kernels()}
    exec(compile(readme_snippet(), "README.md", "exec"), ns)
    return ns["residual_add_rmsnorm"]


def _with_triton(module_name):
    """Import ``module_name`` (a module of the gpu tests), with a stand-in for Triton where Triton is missing."""
    import importlib

    try:
        import triton  # noqa: F401
    except ImportError:
        pass
    else:
        return importlib.import_module(module_name)
    standin = types.ModuleType("triton")
    standin.jit = lambda fn: fn
    language = types.ModuleType("triton.language")
    language.constexpr = type("constexpr", (), {})
    standin.language = language
    sys.modules["triton"], sys.modules["triton.language"] = standin, language
    try:
        return importlib.import_module(module_name)
    finally:
        del sys.modules["triton"], sys.modules["triton.language"]
        sys.modules.pop(module_name, None)  # a later import, with Triton, must not find the stand-in's functions


def gpu_kernels():
    """The ``gpu_kernels`` module of the gpu tests."""
    return _with_triton("gpu_kernels")


def gpu_kernels_p1():
    """The hand-written kernels of the strided-view and device-scalar tests (``gpu_kernels_p1.py``)."""
    return _with_triton("gpu_kernels_p1")


def mm_kernels():
    """The ``tl.dot`` matrix product (``mm_kernels.py``)."""
    return _with_triton("mm_kernels")


def gpu_kernels_p5():
    """The hand-written kernels of the plan features of M3.3 (``gpu_kernels_p5.py``)."""
    return _with_triton("gpu_kernels_p5")


def gpu_kernels_p4():
    """The hand-written kernels of the features of M3.3 (``gpu_kernels_p4.py``)."""
    return _with_triton("gpu_kernels_p4")


def types_kernels():
    """The kernels of 64-bit dimension scalars, null pointers and element-aligned specialisations (``types_kernels.py``)."""
    return _with_triton("types_kernels")


# -- prebuilt PTX that is checked in: hand-written (ptx/), so nothing here needs a toolchain -------------------------------------------------

_N = Dim("n")
_R = Dim("R")


@tilefish.extern(ptx=PTX / "axpy_f32.ptx", symbol="axpy_f32", block=(128,), grid=lambda n: (tilefish.cdiv(n, 128),), name="axpy_ptx")
def axpy_ptx(x: Tensor[_N, f32], y: Tensor[_N, f32], out: Out[Tensor[_N, f32]], n_elem: _N, alpha: const[f32] = 2.0): ...


@tilefish.extern(ptx=PTX / "strided_copy_bf16.ptx", symbol="strided_copy_bf16", block=(64,), grid=lambda R: (R,), index_bits=64)
def strided_copy(x: Strided[Tensor[_R, 64, bf16]], y: Out[Tensor[_R, 64, bf16]], *, cols: const[i32] = 64, x_s0: StrideOf("x", 0)): ...


def ops():
    """``{name: operator}`` in a fixed order: the lowered kernels, the bucketed one, the hand-written externs, then schema 3's."""
    kernels = gpu_kernels()
    p1 = gpu_kernels_p1()
    p4 = gpu_kernels_p4()
    p5 = gpu_kernels_p5()
    tk = types_kernels()
    return {
        "softmax": softmax,
        "layernorm": layernorm,
        "residual_add_rmsnorm": readme_kernel(),
        "gelu_exact": gelu_exact,
        "swiglu": swiglu,
        "bucketed_rms": bucketed_rms,
        "rmsnorm": kernels.rmsnorm,
        "gelu_erf": kernels.gelu_erf,
        "axpy": kernels.axpy,
        "mm": mm_kernels().mm,  # a tl.dot kernel: more than 48 KiB of dynamic shared memory and a 2-D grid
        # strided views
        "head_copy": p1.head_copy,
        "head_copy16": p1.head_copy16,
        "head_copy64": p1.head_copy64,
        "softmax3d": p1.softmax3d,
        # device scalars
        "masked_rows": p1.masked_rows,
        "scale_by": p1.scale_by,
        "scale_flat": p1.scale_flat,
        # prebuilt PTX (the stem of the first is not `axpy`: that is the Triton kernel of gpu_kernels)
        "axpy_ptx": axpy_ptx,
        "strided_copy": strided_copy,
        # an extern's contraction (M3.3)
        "fma_plain": p4.fma_plain,
        "fma_fused": p4.fma_fused,
        # strides that several views share (M3.3)
        "qkv_sum": p4.qkv_sum,
        "qkv_sum16": p4.qkv_sum16,
        # a dimension only the Out tensor determines (M3.3)
        "key_mask": p4.key_mask,
        "key_mask_any": p4.key_mask_any,
        # an Out that may be written over an input (M3.3)
        "add_into": p5.add_into,
        # a 64-bit dimension scalar, a null pointer, both in a prebuilt binary (M3.5)
        "add_one_wide": tk.add_one_wide,
        "scale_null": tk.scale_null,
        "scale_bias": tk.scale_bias,
        "scale_null_ptx": tk.scale_null_ptx,
        "scale_bias_ptx": tk.scale_bias_ptx,
        # the specialisation of a lowered kernel for a first pointer aligned to its element only (M3.5)
        "add_rows_ua": tk.add_rows.unaligned("a"),
    }


def make_gemm(name, n, k, bias):
    """A pinned cuBLASLt GEMM over ``M`` buckets (``test_gpu_gemm.py: make_gemm``)."""
    Mb = Dim("M", buckets=GEMM_BUCKETS)
    if bias:

        @tilefish.library(name, acc=f32, impl="cublaslt.gemm_bias")
        def gemm_bias(a: Tensor[Mb, k, bf16], b: Tensor[n, k, bf16], bias: Tensor[n, bf16]) -> Tensor[Mb, n, bf16]: ...

        return gemm_bias

    @tilefish.library(name, acc=f32, impl="cublaslt.gemm")
    def gemm(a: Tensor[Mb, k, bf16], b: Tensor[n, k, bf16]) -> Tensor[Mb, n, bf16]: ...

    return gemm


def gluon_ops():
    """``{name: extern}``: the Gluon kernels of ``gluon_kernels.py`` (which imports ``triton.experimental.gluon``)."""
    import gluon_kernels

    return {"row_dot": gluon_kernels.row_dot, "row_dot_fma": gluon_kernels.row_dot_fma}


def per_arch_ops():
    """``{name: extern}``: the op of ``per_arch_kernels.py`` with an implementation per architecture (a Triton kernel each)."""
    import per_arch_kernels

    return {"row_sum": per_arch_kernels.row_sum}


def generated_ops():
    """``{name: extern}``: generated ops (Triton tilefish writes). Of ``tilefish.im2col1d``, one per column order: ``im2col1d_tap``, a causal window
    of 7 taps at dilation 3 over 256 static channels (what a channel-last activation permuted to ``[N, C, L]`` gives), and ``im2col1d_channel``, 3
    taps at stride 2 with a padding of ``(2, 5)`` and a runtime channel count. Of ``tilefish.col2im1d``: ``col2im1d_channel``, f32 in the channel
    order to ``[N, C, Lo]`` with a bias (4 taps at stride 2, padding 1: an upsampling by 2) and a runtime channel count, and ``col2im1d_tap``, bf16 in
    the tap order to a channel-last ``[N, Lo, C]`` (7 taps at stride 3, dilation 2, padding ``(2, 5)``, output padding 1) over 64 static channels.
    Their stems are their names; the committed fixtures do not have them."""
    from tilefish._copyops import col2im1d_op, im2col1d_op

    return {
        "im2col1d_tap": im2col1d_op(bf16, 7, dilation=3, padding=(18, 0), order="tap", channels=256),
        "im2col1d_channel": im2col1d_op(bf16, 3, stride=2, padding=(2, 5)),
        "col2im1d_channel": col2im1d_op(f32, 4, stride=2, padding=1, bias=True),
        "col2im1d_tap": col2im1d_op(bf16, 7, stride=3, dilation=2, padding=(2, 5), output_padding=1, order="tap", layout="nlc", channels=64),
    }


def broadcast_ops():
    """``{name: kernel}``: lowered kernels of ``broadcast_cases.py`` that read an operand broadcast along an axis that is not a leading one, whose
    binaries take the sizes the lane's coordinates need as runtime scalars of the dimensions: ``key_mask_heads`` (a ``[4, 1, L]`` mask over
    ``[4, L, L]`` scores, whose stride is ``L*L``), ``middle_rows`` (``[K, 1, D]`` over ``[K, T, D]``) and ``causal_heads`` (an ``iota`` along two axes).
    The committed fixtures do not have them."""
    import broadcast_cases as bc

    return {"key_mask_heads": bc.key_mask_heads, "middle_rows": bc.middle_rows, "causal_heads": bc.causal_heads}


def ieee_ops():
    """``{name: kernel}``: lowered kernels in IEEE math (``ieee_cases.py``, note 13 of ``tilefish.lower``), whose binaries hold inline PTX:
    ``math_ieee`` (CUDA's ``expf``, ``sqrtf``, ``rsqrtf``, ``sinf`` and torch's silu), ``div_ieee`` (``div.rn.f32``) and ``periodic`` (a row kernel in
    IEEE math without contraction, one binary per bucket of its columns). The committed fixtures do not have them."""
    import ieee_cases as ic

    return {"math_ieee": ic.math_ieee, "div_ieee": ic.div_ieee, "periodic": ic.periodic}


def tanh_ops():
    """``{name: kernel}``: lowered kernels of ``tilefish.tanh`` (``tanh_cases.py``, note 14 of ``tilefish.lower``), whose binaries inline libdevice's
    ``tanhf``: ``tanh_fast`` (built with flush to zero) and ``tanh_rows`` (a bias and a tanh of bf16 rows in IEEE math). The committed fixtures do
    not have them."""
    import tanh_cases as tc

    return {"tanh_fast": tc.tanh_fast, "tanh_rows": tc.tanh_rows}


def primitive_ops():
    """``{name: kernel}``: lowered kernels of the primitives of notes 15 to 17 of ``tilefish.lower`` (``primitive_cases.py``): CUDA's ``erff`` in IEEE
    math (an entry with ``cuda_math``), an ``fma``, CUDA's negation, and an rmsnorm whose sum of squares is an ordered ``fma`` sum. The committed
    fixtures do not have them."""
    import primitive_cases as pc

    return {"erf_ieee": pc.erf_ieee, "fma3": pc.fma3, "neg_f32": pc.neg_f32, "rmsnorm_ordered": pc.rmsnorm_ordered}


def gemm_name(n, k, bias):
    return f"gemm_{n}_{k}{'_bias' if bias else ''}"


def library_ops(shapes=GEMM_SHAPES):
    """``{name: library op}``: ``gemm`` and ``gemm_bias`` for each ``(N, K)`` of ``shapes``, each over the buckets ``GEMM_BUCKETS`` of ``M``.

    The names are ``gemm_<N>_<K>`` and ``gemm_<N>_<K>_bias`` and the stems ``<name>.M<bucket>`` (so the struct of the Rust launcher of
    ``gemm_1024_1024.M32`` is ``Gemm10241024M32``). They are pinned on a GPU (``tilefish.runtime.export``), for the architecture of the device.
    """
    out = {}
    for n, k in shapes:
        for bias in (False, True):
            name = gemm_name(n, k, bias)
            out[name] = make_gemm(name, n, k, bias)
    return out


# -- what needs the GPU toolchain to be declared ----------------------------------------------------------------------------------------------


def _triton_ptx(lowered, directory, arch, name, reuse):
    """The Triton build of a lowered kernel for ``arch`` and its PTX file in ``directory`` (what Triton hands to ptxas): ``(path, build)``.

    The file is written from the build, unless ``reuse`` and it is there already (then that file is the one that is declared)."""
    from tilefish._build import build
    from tilefish.aot import compile_specs
    from tilefish.runtime import load_lowered

    (spec,) = compile_specs(lowered.extern)
    built = build(spec, arch=arch, fn=load_lowered(lowered), cache=False)
    path = pathlib.Path(directory) / f"{name}.sm_{arch}.ptx"
    if not (reuse and path.exists()):
        path.write_text(built.ptx)
    return path, built


def gpu_ops(directory, arch, *, reuse=False):
    """``{name: operator}`` of the prebuilt externs that need a GPU toolchain (Triton and ptxas) to be declared, for ``arch`` (a capability, 103).

    The files go to ``directory`` (created): ``gelu_ptx.sm_<arch>.ptx`` and ``softmax_ptx.sm_<arch>.ptx`` (the PTX of Triton's builds of the
    lowered ``gelu_exact`` and ``softmax``, declared with Triton's ABI and its two scratch slots as fixed ``i64`` arguments), and
    ``axpy_cubin.sm_<arch>.cubin`` (ptxas of ``ptx/axpy_f32.ptx``). A declaration reads its file once, so the ops are made together with the
    files. With ``reuse=True`` a file that is there is declared as it is (what a gate that runs against an already exported manifest does:
    its binaries are what the manifest was made from); otherwise each is made again, so an export shows that a fresh Triton build and
    its PTX still give the same bytes.
    """
    import subprocess

    from tilefish import _binary

    directory = pathlib.Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    gelu_path, gelu_built = _triton_ptx(tilefish.lower(gelu_exact), directory, arch, "gelu_ptx", reuse)
    softmax_path, softmax_built = _triton_ptx(tilefish.lower(softmax), directory, arch, "softmax_ptx", reuse)

    @tilefish.extern(
        ptx=gelu_path,
        symbol=gelu_built.symbol,
        block=(gelu_built.num_warps * 32,),
        shared=gelu_built.shared,
        grid=lambda T: (tilefish.cdiv(T, 1024),),
        name="gelu_ptx",
    )
    def gelu_ptx(x: Tensor[T, bf16], out: Out[Tensor[T, bf16]], n_: T, global_scratch: const[i64] = 0, profile_scratch: const[i64] = 0): ...

    Mrows = Dim("M")

    @tilefish.extern(
        ptx=softmax_path,
        symbol=softmax_built.symbol,
        block=(softmax_built.num_warps * 32,),
        shared=softmax_built.shared,
        grid=lambda M: (M,),
        name="softmax_ptx",
    )
    def softmax_ptx(x: Tensor[Mrows, 1000, f32], out: Out[Tensor[Mrows, 1000, f32]], global_scratch: const[i64] = 0, profile_scratch: const[i64] = 0): ...

    cubin = directory / f"axpy_cubin.sm_{arch}.cubin"
    if not (reuse and cubin.exists()):
        done = subprocess.run(
            [_binary.find_ptxas(), *_binary.PTX_FLAGS, f"--gpu-name=sm_{arch}", str(PTX / "axpy_f32.ptx"), "-o", str(cubin)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if done.returncode != 0:
            raise RuntimeError(f"ptxas could not make the cubin of axpy_cubin for sm_{arch}: {done.stderr[-2000:]}")

    @tilefish.extern(cubin={f"sm_{arch}": cubin}, symbol="axpy_f32", block=(128,), grid=lambda n: (tilefish.cdiv(n, 128),), name="axpy_cubin")
    def axpy_cubin(x: Tensor[_N, f32], y: Tensor[_N, f32], out: Out[Tensor[_N, f32]], n_elem: _N, alpha: const[f32] = 2.0): ...

    return {"gelu_ptx": gelu_ptx, "softmax_ptx": softmax_ptx, "axpy_cubin": axpy_cubin}


def abi_of(op):
    """``(tensors, inputs, outputs)``: the tensor parameters of the launched kernel in ABI order with their access, the
    names of the arguments of a call (``In`` and ``InOut``) and of the results of a call, in order."""
    target = runtime._target(op)
    extern = target.launch_extern()
    names = runtime._extern_names(target, extern)
    tensors = [(p.name, p.access) for p in extern.signature.params if p.kind == "tensor"]
    if target.kernel is None:
        inputs = [n for n, a in tensors if a != "out"]
        outputs = [n for n, a in tensors if a == "out"]
    else:
        user = len(target.kernel.signature.params)
        returned = [p.name for p in extern.signature.params[user:] if p.kind == "tensor" and p.access == "out"]
        outputs = returned + [names[p.name] for p in target.kernel.signature.tensor_outputs]
        inputs = [names[p.name] for p in target.kernel.signature.tensor_inputs]
    return tensors, inputs, outputs
