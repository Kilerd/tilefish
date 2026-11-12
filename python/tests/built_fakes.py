"""Two externs and a ``BuiltKernel`` made by hand, for the tests that need a build without Triton (not collected).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import hashlib

from elf_builder import cubin_like

import tilefish
from tilefish import Dim, Out, Tensor, bf16, const, f32
from tilefish._build import LaunchAttributes
from tilefish._elf import code_digest
from tilefish.aot import BuiltKernel

M = Dim("M")
S = Dim("S", buckets=[32, 64])
n = Dim("n", divisible_by=16)


@tilefish.extern(triton="my_kernels:scale", num_warps=4, grid=lambda n: (tilefish.cdiv(n, 256),))
def scale(x: Tensor[n, bf16], y: Out[Tensor[n, bf16]], n_elem: n, *, ALPHA: const[f32] = float("inf"), BLOCK: const = 256): ...


@tilefish.extern(triton="my_kernels:rowop", num_warps=2, grid=lambda M: (M,))
def rowop(x: Tensor[M, S, f32], y: Out[Tensor[M, S, f32]], *, EPS: const[f32] = 1e-5, BLOCK: const = tilefish.dims.next_pow2(S)): ...


def fake_build(spec, *, arch="sm_90", cubin=None, launch=None, **kw):
    """A ``BuiltKernel`` of ``spec`` around a small ELF file (its symbol is ``tf_k``); ``kw`` overrides any field."""
    cubin = cubin_like() if cubin is None else cubin
    values = dict(
        spec=spec,
        arch=arch,
        cubin=cubin,
        ptx=".version 8.8\n.target sm_90a\n",
        symbol="tf_k",
        shared=0,
        num_warps=spec.num_warps,
        sha256=hashlib.sha256(cubin).hexdigest(),
        sha256_code=code_digest(cubin),
        triton_version="3.5.1",
        ptxas_version="Cuda compilation tools, release 12.9, V12.9.86",
        ptx_target="sm_90a",
        options=spec.ast_source_inputs().options,
        launch=launch or LaunchAttributes(0, 0, False, (1, 1, 1), 1, False),
    )
    values.update(kw)
    return BuiltKernel(**values)
