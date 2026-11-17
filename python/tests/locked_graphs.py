"""The graphs of the locked export's gate: a plan set with a step of every kind of callee an export builds or pins (not collected: the name has no
``test_`` prefix). ``docs/guide/locked-export.md``; ``test_gpu_locked_export.py`` and ``rust_gate.py prepare --locked`` export it.

Importing this module imports Triton (and its Gluon) and runs ptxas (the cubin extern is the PTX ``ptx/axpy_f32.ptx`` assembled here, the same bytes on
every machine with the same ptxas), so only the gpu tests, the gate and a machine that makes the locked export do. Written for these tests, and public.
No ``from __future__ import annotations``: tilefish evaluates annotations.

* ``lk_mlp`` (``M`` in 32 and 64): a generated kernel (``lk_rmsnorm``, a lowered ``@tilefish.kernel``), a pinned cuBLASLt GEMM with a bias
  (``lk_up``, its workspace in the arena), the arithmetic of the body (a SiLU between two casts: a fused region, which an export verifies on the
  device), a cuBLASLt matmul with an epilogue (``lk_down``: ``relu((acc + c) + bias)``, ``c`` the graph's input; its pin verified at the export and
  a canary), and a hand-written Triton extern (``lk_scale``).
* ``lk_conv`` (``T`` in 64 and 128): a pinned cuDNN convolution (``lk_conv1d``, NHWC, the backend API, a canary), the body's arithmetic (a cast and a
  product: a fused region), a Gluon extern over two views (``gluon_kernels.row_dot``), a prebuilt PTX (``lk_axpy_ptx``) and a prebuilt cubin
  (``lk_axpy_cubin``) over a flat view, and a Triton extern whose device-side TMA descriptors live in its global scratch, the plan's workspace
  (``launch_kernels.desc_scale``).
"""

import pathlib
import subprocess
import tempfile

import gluon_kernels as gk
import launch_kernels as lk
import triton
import triton.language as tl

import tilefish
from tilefish import Dim, Out, Tensor, bf16, const, f16, f32
from tilefish import _binary

HERE = pathlib.Path(__file__).resolve().parent
PTX = HERE / "ptx" / "axpy_f32.ptx"
CUBIN_ARCHES = ("sm_90", "sm_100", "sm_103", "sm_120")  # the cubin extern has a cubin for each (a cubin is used as it is, for its architecture)

M = Dim("M", buckets=[32, 64])
T = Dim("T", buckets=[64, 128])
n = Dim("n")


# -- lk_mlp ---------------------------------------------------------------------------------------------------------------------------------


@tilefish.kernel
def lk_rmsnorm(x: Tensor[M, 256, bf16], w: Tensor[256, bf16], *, eps: const[f32] = 1e-6) -> Tensor[M, 256, bf16]:
    xf = x.to(f32)
    ms = tilefish.mean(xf * xf, axis=-1, keepdims=True)
    return (xf * tilefish.rsqrt(ms + eps) * w.to(f32)).to(bf16)


@tilefish.library("lk_up", acc=f32, impl="cublaslt.gemm_bias")
def lk_up(a: Tensor[M, 256, bf16], b: Tensor[512, 256, bf16], bias: Tensor[512, bf16]) -> Tensor[M, 512, bf16]: ...


@tilefish.library("lk_down", acc=f32, impl="cublaslt.matmul", epilogue="relu_bias", beta=1)
def lk_down(a: Tensor[M, 512, bf16], b: Tensor[256, 512, bf16], bias: Tensor[256, bf16], c: Tensor[M, 256, bf16]) -> Tensor[M, 256, bf16]: ...


@triton.jit
def _lk_scale(x, y, COLS: tl.constexpr):
    cols = tl.arange(0, COLS)
    at = tl.program_id(0) * COLS + cols
    tl.store(y + at, (tl.load(x + at).to(tl.float32) * 2.0).to(tl.bfloat16))


@tilefish.extern(triton=_lk_scale, num_warps=4, grid=lambda M: (M,), reference=lambda x: (x.float() * 2).to(x.dtype))
def lk_scale(x: Tensor[M, 256, bf16], y: Out[Tensor[M, 256, bf16]], *, COLS: const = 256): ...


@tilefish.graph(inputs=("x",), weights=("w_n", "w_up", "b_up", "w_down", "b_down"))
def lk_mlp(
    x: Tensor[M, 256, bf16],
    w_n: Tensor[256, bf16],
    w_up: Tensor[512, 256, bf16],
    b_up: Tensor[512, bf16],
    w_down: Tensor[256, 512, bf16],
    b_down: Tensor[256, bf16],
):
    h = lk_rmsnorm(x, w_n)
    u = lk_up(h, w_up, b_up)
    g = tilefish.silu(u.to(f32)).to(bf16)  # a fused region of the body
    y = lk_down(g, w_down, b_down, x)  # relu((g @ w_down^T + x) + b_down), rounded once
    return lk_scale(y)


# -- lk_conv --------------------------------------------------------------------------------------------------------------------------------


@tilefish.library("lk_conv1d", impl="cudnn.conv_fwd", layout="NHWC", padding=1)
def lk_conv1d(x: Tensor[1, T, 128, f16], w: Tensor[128, 3, 128, f16]) -> Tensor[1, T, 128, f16]: ...


@tilefish.extern(ptx=PTX, symbol="axpy_f32", block=(128,), grid=lambda n: (tilefish.cdiv(n, 128),), name="lk_axpy_ptx")
def lk_axpy_ptx(x: Tensor[n, f32], y: Tensor[n, f32], out: Out[Tensor[n, f32]], n_elem: n, alpha: const[f32] = 2.0): ...


def _assemble() -> "dict[str, pathlib.Path]":
    """``ptx/axpy_f32.ptx`` assembled for each of :data:`CUBIN_ARCHES` the ptxas a build uses can target (``tilefish._binary.find_ptxas``; Triton
    3.5.1's own, CUDA 12.8's, has no sm_103), with the flags a prebuilt PTX is assembled with, from a directory of its own (so the command line
    ptxas records names no path of this machine). The same ptxas on two machines gives the same cubins, so the same declaration."""
    out = {}
    root = pathlib.Path(tempfile.mkdtemp(prefix="tilefish-locked-"))
    (root / "axpy_f32.ptx").write_bytes(PTX.read_bytes())
    for arch in CUBIN_ARCHES:
        name = f"axpy_f32.{arch}.cubin"
        done = subprocess.run([_binary.find_ptxas(), *_binary.PTX_FLAGS, f"--gpu-name={arch}", "axpy_f32.ptx", "-o", name], cwd=root, capture_output=True)
        if done.returncode == 0:
            out[arch] = root / name
    return out


CUBINS = _assemble()  # the architectures the set exports for: those the cubin extern has a cubin of


@tilefish.extern(cubin=CUBINS, symbol="axpy_f32", block=(128,), grid=lambda n: (tilefish.cdiv(n, 128),), name="lk_axpy_cubin")
def lk_axpy_cubin(x: Tensor[n, f32], y: Tensor[n, f32], out: Out[Tensor[n, f32]], n_elem: n, alpha: const[f32] = 0.5): ...


@tilefish.graph(inputs=("x",), weights=("w",))
def lk_conv(x: Tensor[1, T, 128, f16], w: Tensor[128, 3, 128, f16]):
    z = lk_conv1d(x, w).reshape(T, 128).to(f32) * 0.5  # a fused region of the body
    d = gk.row_dot(z, z)  # Gluon: a row dot of two views
    flat = z.reshape(T * 128)
    a = lk_axpy_ptx(flat, flat)  # 2 z + z
    b = lk_axpy_cubin(a, flat)  # a / 2 + z
    s = lk.desc_scale(z.reshape(T * 2, 64))  # device-side TMA descriptors in the plan's workspace
    return d, b, s


GRAPHS = (lk_mlp, lk_conv)


def cases(graph):
    """``(label, {parameter: tensor})`` of each bucket of ``graph``, two seeds each; the weights are the same tensors in every case."""
    import torch
    from gpu_util import randn

    if graph is lk_mlp:
        weights = {
            "w_n": randn((256,), torch.bfloat16, seed=300, scale=0.5, shift=1.0),
            "w_up": randn((512, 256), torch.bfloat16, seed=301, scale=256**-0.5),
            "b_up": randn((512,), torch.bfloat16, seed=302),
            "w_down": randn((256, 512), torch.bfloat16, seed=303, scale=512**-0.5),
            "b_down": randn((256,), torch.bfloat16, seed=304),
        }
        return [(f"M{m}-seed{seed}", {"x": randn((m, 256), torch.bfloat16, seed=310 + m + seed), **weights}) for m in M.buckets for seed in (0, 1)]
    w = randn((128, 3, 128), torch.float16, seed=320, scale=(3 * 128) ** -0.5)
    return [(f"T{t}-seed{seed}", {"x": randn((1, t, 128), torch.float16, seed=330 + t + seed), "w": w}) for t in T.buckets for seed in (0, 1)]
