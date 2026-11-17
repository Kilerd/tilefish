"""CuTe DSL kernels of the public CUTLASS examples, declared as tilefish externs (not collected: the name has no ``test_`` prefix).

Needs the CuTe DSL (``nvidia-cutlass-dsl``) and the examples of a CUTLASS checkout: ``TILEFISH_CUTLASS_EXAMPLES`` is its
``examples/python/CuTeDSL`` directory (the examples are BSD-3-Clause and are not copied into this repository; ``docs/en/cute.md`` says which release
the gate ran). Each example module is loaded under a name of its own (two of them are ``dense_gemm.py``).

* ``gemm``: ``C = A @ B.T`` in f16 with an f32 accumulator, ``A`` of ``[M, K]``, ``B`` of ``[N, K]``: on sm_100 and sm_103 the persistent tcgen05 GEMM
  of ``cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py`` (2-CTA MMA 256x128, a cluster of 2x1, TMA loads and a TMA store: three TMA
  descriptors the DSL encodes, and the fast divisors of its tile scheduler, structures with padding the kernel does not read), on sm_120 the warp-MMA
  GEMM of ``cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py``; both persistent, their grid the clusters the device runs at once when it is
  built. One operator, an implementation per architecture.
* ``gemm_dense``: the same product by the non-persistent tcgen05 GEMM of ``cute/blackwell/kernel/dense_gemm/dense_gemm.py`` (sm_100 and sm_103: a
  grid of one cluster per output tile of 256x128).
* ``add`` and ``leaky_relu``: the elementwise kernel of ``cute/ampere/kernel/elementwise/elementwise_apply.py`` with ``operator.add`` and with its
  ``leaky_relu`` (f32: its alpha is an f32), pointers and constant bytes.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import functools
import importlib.util
import operator
import os
import pathlib
import sys

import cutlass

import tilefish
from tilefish import Dim, Out, Tensor, f16, f32

EXAMPLES = pathlib.Path(os.environ["TILEFISH_CUTLASS_EXAMPLES"])


def _load(name, relative):
    path = EXAMPLES / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sm100 = _load("cutlass_example_dense_gemm_sm100", "cute/blackwell/kernel/dense_gemm/dense_gemm.py")
sm100_persistent = _load("cutlass_example_dense_gemm_persistent_sm100", "cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py")
sm120 = _load("cutlass_example_dense_gemm_sm120", "cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py")
elementwise = _load("cutlass_example_elementwise_apply", "cute/ampere/kernel/elementwise/elementwise_apply.py")

M = Dim("M", buckets=[256, 1024, 4096])
N, K = 2048, 1024
R = Dim("R", buckets=[64, 4096])
C = 1024


def batched(a, b, c):
    """The ``(M, K, L)`` tensors of the examples' GEMMs, ``L = 1``: views of the operator's tensors."""
    return a.unsqueeze(-1), b.unsqueeze(-1), c.unsqueeze(-1)


def persistent_args(cluster):
    """A persistent GEMM's grid is the clusters the device runs at once, a constexpr of the host function (for the device it is built on)."""

    def make(a, b, c):
        return (*batched(a, b, c), cutlass.utils.HardwareInfo().get_max_active_clusters(cluster))

    return make


TCGEN05 = tilefish.impl(
    cute=sm100_persistent.PersistentDenseGemmKernel(cutlass.Float32, True, (256, 128), (2, 1), True), cute_args=persistent_args(2), name="tcgen05"
)


@functools.lru_cache(maxsize=None)
def _sm120_gemm():
    return sm120.Sm120GemmKernel(cutlass.Float32, (128, 128, 64))


@tilefish.extern(by_arch={"sm_100": TCGEN05, "sm_103": TCGEN05, "sm_120": tilefish.impl(cute=_sm120_gemm(), cute_args=persistent_args(1))}, name="cute_gemm")
def gemm(a: Tensor[M, K, f16], b: Tensor[N, K, f16], c: Out[Tensor[M, N, f16]]): ...


@tilefish.extern(cute=sm100.DenseGemmKernel(cutlass.Float32, True, (256, 128), (2, 1), True), cute_args=batched, name="cute_gemm_dense")
def gemm_dense(a: Tensor[M, K, f16], b: Tensor[N, K, f16], c: Out[Tensor[M, N, f16]]): ...


@tilefish.extern(cute=elementwise.elementwise_apply, cute_args=lambda a, b, c: (operator.add, [a, b], c), name="cute_add")
def add(a: Tensor[R, C, f16], b: Tensor[R, C, f16], c: Out[Tensor[R, C, f16]]): ...


@tilefish.extern(cute=elementwise.elementwise_apply, cute_args=lambda a, c: (functools.partial(elementwise.leaky_relu, alpha=0.01), [a], c), name="cute_leaky_relu")
def leaky_relu(a: Tensor[R, C, f32], c: Out[Tensor[R, C, f32]]): ...


@tilefish.extern(cute=elementwise.elementwise_apply, cute_args=lambda a, b, c: (operator.add, [a, b], c), name="cute_add_mn")
def add_mn(a: Tensor[M, N, f16], b: Tensor[M, N, f16], c: Out[Tensor[M, N, f16]]): ...


@tilefish.graph(name="cute_chain")
def gemm_then_add(a: Tensor[M, K, f16], b: Tensor[N, K, f16], r: Tensor[M, N, f16]) -> Tensor[M, N, f16]:
    return add_mn(gemm(a, b), r)


def ops(capability=None):
    """``{name: op}`` of every CuTe DSL extern here a device of ``capability`` (``(10, 3)``; ``None``: any) runs: what the Rust gate exports. The
    names (``cute_gemm``, ...) are the operators', so the stems are ``cute_gemm.M256``, ..."""
    out = {"cute_add": add, "cute_leaky_relu": leaky_relu, "cute_add_mn": add_mn}
    if capability is None or tuple(capability) in ((10, 0), (10, 3), (12, 0)):
        out["cute_gemm"] = gemm
    if capability is None or tuple(capability) in ((10, 0), (10, 3)):
        out["cute_gemm_dense"] = gemm_dense
    return out
