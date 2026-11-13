"""A hand-written ``tl.dot`` matrix product for the Rust host's gate (not collected: the name has no ``test_`` prefix).

It is the kernel of ``test_gpu_launch_limits.py`` written as a module of its own (that test writes it to a temporary directory),
declared with ``tilefish.extern``: 8 warps and 4 stages of 128 x 64 f16 tiles need about 128 KiB of dynamic shared memory, which a
launch asks the driver for with ``cuFuncSetAttribute`` (above the default 48 KiB), and the grid is two-dimensional. Its dimensions are
declared ``divisible_by`` 128, 128 and 64, which the manifest records and a host checks.

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import triton
import triton.language as tl

import tilefish
from tilefish import Dim, Out, Tensor, const, f16, f32


@triton.jit
def mm_kernel(a_ptr, b_ptr, c_ptr, m, n, k, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, k, BK):
        a = tl.load(a_ptr + offs_m[:, None] * k + (kk + offs_k)[None, :])
        b = tl.load(b_ptr + (kk + offs_k)[:, None] * n + offs_n[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], acc)


MmM, MmN, MmK = Dim("MmM", divisible_by=128), Dim("MmN", divisible_by=128), Dim("MmK", divisible_by=64)


@tilefish.extern(
    triton=mm_kernel, num_warps=8, num_stages=4, grid=lambda MmM, MmN: (tilefish.cdiv(MmM, 128), tilefish.cdiv(MmN, 128))
)
def mm(
    a_ptr: Tensor[MmM, MmK, f16],
    b_ptr: Tensor[MmK, MmN, f16],
    c_ptr: Out[Tensor[MmM, MmN, f32]],
    m: MmM,
    n: MmN,
    k: MmK,
    *,
    BM: const = 128,
    BN: const = 128,
    BK: const = 64,
): ...
