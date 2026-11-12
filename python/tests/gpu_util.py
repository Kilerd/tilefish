"""Helpers of the gpu tests (not collected: the name has no ``test_`` prefix). Needs torch."""

import torch

DEVICE = "cuda"

_SIGNED = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}


def randn(shape, dtype=torch.float32, seed=0, scale=1.0, shift=0.0):
    """Normal random values on the GPU, from a generator of their own (the same on every machine)."""
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    values = torch.randn(shape, generator=gen, device=DEVICE, dtype=torch.float32) * scale + shift
    return values.to(dtype)


def bits(t):
    """The raw bytes of a tensor, as a uint8 tensor."""
    return t.contiguous().view(torch.uint8)


def same_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b))


def differing(a, b):
    """The number of elements whose bytes differ."""
    assert a.dtype == b.dtype and a.shape == b.shape
    width = a.element_size()
    return int((bits(a).reshape(-1, width) != bits(b).reshape(-1, width)).any(-1).sum())


def ordinal(t):
    """A monotonic integer image of a float tensor: the distance of two images is the distance in representable values."""
    signed = _SIGNED[t.dtype]
    n = t.element_size() * 8
    i = t.contiguous().view(signed).to(torch.int64)
    magnitude = i & ((1 << (n - 1)) - 1)
    return torch.where(i < 0, -magnitude, i)


def error(actual, expected):
    """``(max ULP, max abs error, number of differing elements, number of elements)`` of two float tensors of one dtype."""
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert not torch.isnan(actual).any() and not torch.isnan(expected).any()
    ulp = (ordinal(actual) - ordinal(expected)).abs()
    absolute = (actual.double() - expected.double()).abs()
    return int(ulp.max()), float(absolute.max()), int((ulp > 0).sum()), actual.numel()
