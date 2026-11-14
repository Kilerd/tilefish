"""cuDNN convolutions (``@tilefish.library(..., impl="cudnn.conv_fwd")``) shared by the GPU tests and the Rust gate: both APIs, both layouts, one and
two spatial axes, f16, bf16 and f32, a stride, a dilation, groups, a single input channel, and a plan of two of them."""

import tilefish
from tilefish import Dim, Tensor, bf16, cdiv, f16, f32

T = Dim("T", buckets=[64, 646])

# a 1-D NHWC convolution of 64 channels, k = 7, the same length (padding 3): the backend API under torch's posture
@tilefish.library("conv_k7", impl="cudnn.conv_fwd", layout="NHWC", padding=3)
def conv_k7(x: Tensor[1, T, 64, f16], w: Tensor[64, 7, 64, f16]) -> Tensor[1, T, 64, f16]: ...


# the same with a dilation of 3 (padding 9), through the legacy API (cudnnGetConvolutionForwardAlgorithm_v7's first deterministic answer)
@tilefish.library("conv_k7_dil", impl="cudnn.conv_fwd", layout="NHWC", padding=9, dilation=3, api="legacy")
def conv_k7_dil(x: Tensor[1, T, 64, f16], w: Tensor[64, 7, 64, f16]) -> Tensor[1, T, 64, f16]: ...


# a strided down-sampling convolution, 64 -> 128 channels, k = 4, stride 2: half the length
@tilefish.library("conv_down", impl="cudnn.conv_fwd", layout="NHWC", stride=2, padding=1)
def conv_down(x: Tensor[1, T, 64, f16], w: Tensor[128, 4, 64, f16]) -> Tensor[1, cdiv(T, 2), 128, f16]: ...


# one input channel, k = 7, the symmetric padding of a causal convolution (padding 6: the first T of T + 6 rows are its output), legacy
@tilefish.library("conv_in", impl="cudnn.conv_fwd", layout="NHWC", padding=6, api="legacy")
def conv_in(x: Tensor[1, T, 1, f16], w: Tensor[64, 7, 1, f16]) -> Tensor[1, T + 6, 64, f16]: ...


# NCHW bf16, depthwise (groups = channels), as torch's conv1d takes it
@tilefish.library("conv_dw", impl="cudnn.conv_fwd", padding=3, groups=96)
def conv_dw(x: Tensor[2, 96, 500, bf16], w: Tensor[96, 1, 7, bf16]) -> Tensor[2, 96, 500, bf16]: ...


# NCHW f32 2-D, TF32 off
@tilefish.library("conv_img", impl="cudnn.conv_fwd", padding=1, math="fma")
def conv_img(x: Tensor[2, 16, 24, 24, f32], w: Tensor[32, 16, 3, 3, f32]) -> Tensor[2, 32, 24, 24, f32]: ...


CONVS = {op.name: op for op in (conv_k7, conv_k7_dil, conv_down, conv_in, conv_dw, conv_img)}


@tilefish.graph(inputs=("x",), weights=("w1", "w2"))
def conv_chain(x: Tensor[1, T, 64, f16], w1: Tensor[64, 7, 64, f16], w2: Tensor[64, 7, 64, f16]):
    """Two convolutions, a backend pin and a legacy one: the arena holds the intermediate and the workspace of both."""
    return conv_k7_dil(conv_k7(x, w1), w2)


def operands(op, values, seed=0):
    """Random ``x`` and ``w`` of a declaration at the bucket ``values`` (a dict of dimension name -> size), ``w`` scaled by its fan-in."""
    import torch
    from gpu_util import randn

    from tilefish import _conv

    sig = op.signature
    mapping = {d: values[d.name] for p in sig.params for e in p.type.shape for d in e.free_dims()}
    problem = _conv.problem_at(op.name, sig, op.conv, mapping)
    dtype = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[problem.dtype]
    fan = problem.c // problem.groups
    for r in problem.filter:
        fan *= r
    x = randn(problem.tilefish_shape("x"), dtype, seed=seed)
    w = randn(problem.tilefish_shape("w"), dtype, seed=seed + 1, scale=1 / fan**0.5)
    return x, w
