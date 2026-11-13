"""The operators the Rust host is tested with (not collected: the name has no ``test_`` prefix).

Five lowered kernels (the goldens of the lowering: softmax, layernorm, the README's residual add + rmsnorm with its
``InOut`` tensor, the exact gelu, swiglu), a row kernel with a bucketed row length (one binary per bucket), and three
hand-written Triton kernels declared with ``tilefish.extern``: an rmsnorm, an exact-erf gelu and an in-place
``y += alpha * x`` (``gpu_kernels.py``, and ``mm_kernels.py`` for the matrix product). They are what ``export_rust_fixtures.py`` builds into the fixtures of the crate and what
``rust_gate.py`` runs in Python (``replay``) and in Rust.

``ops()`` works without Triton: ``gpu_kernels`` only needs ``triton.jit`` and ``tl.constexpr`` to be *declared*, and a
declaration (what ``emit`` types a launcher from) does not run a kernel, so where Triton is not installed a stand-in
module is there while it is imported. Its declarations are the real ones: the Triton function is named by its module and
name, and that is all a declaration records of it.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import sys
import types

from conftest import make_my_kernels, readme_snippet
from lower_cases import M, gelu_exact, layernorm, softmax, swiglu

import tilefish
from tilefish import Dim, Tensor, bf16, const, f32, runtime

Db = Dim("Db", buckets=[96, 512, 1000])


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


def mm_kernels():
    """The ``tl.dot`` matrix product (``mm_kernels.py``)."""
    return _with_triton("mm_kernels")


def ops():
    """``{name: operator}`` in a fixed order: the lowered kernels, the bucketed one, then the hand-written externs."""
    kernels = gpu_kernels()
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
    }


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
