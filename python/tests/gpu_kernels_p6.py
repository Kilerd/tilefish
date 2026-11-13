"""Graphs for the gpu tests of the plan features of M3.3 (third part; not collected: the name has no ``test_`` prefix). Synthetic and public.

* ``static_affine``: a graph whose shapes are all static, which declares its roles (``inputs=("x",)``): ``x`` is an input, ``w`` and ``b`` are
  weights, which a host binds once (``tilefish.runtime.bind_weights``, Rust ``PlanRun::bind_weights``). Without the declaration all three would
  be labelled weights.

* ``scale_add``: a prebuilt PTX (``ptx/scale_add_f32.ptx``, ``y = x * k + c``) whose ``k: const[i32]`` and ``c: const[i64]`` have no default: fixed
  arguments given per call. One binary (one manifest entry) serves every value; ``two_scales`` calls it twice, at two values, one of them the
  graph's dimension ``Rg`` (a value per bucket), and a plan records the values in each step.

No ``from __future__ import annotations``: tilefish evaluates annotations.
"""

import pathlib

import tilefish
from tilefish import Dim, Out, Tensor, const, f32, i32, i64

PTX = pathlib.Path(__file__).resolve().parent / "ptx"
Nn = Dim("Nn")
Rg = Dim("Rg", buckets=[8, 24])


@tilefish.kernel
def scale_shift(x: Tensor[16, 64, f32], w: Tensor[16, 64, f32], b: Tensor[16, 64, f32]) -> Tensor[16, 64, f32]:
    return x * w + b


@tilefish.kernel
def double(x: Tensor[16, 64, f32]) -> Tensor[16, 64, f32]:
    return x * 2.0


@tilefish.graph(inputs=("x",))
def static_affine(x: Tensor[16, 64, f32], w: Tensor[16, 64, f32], b: Tensor[16, 64, f32]):
    """``2 * (x * w + b)``: two steps, the weights read by the first."""
    return double(scale_shift(x, w, b))


# -- fixed arguments given per call ---------------------------------------------------------------------------------------------------------


def scale_add_reference(x, k, c):
    return x * float(k) + float(c)  # the product and the sum each rounded to f32, as the PTX does


@tilefish.extern(
    ptx=PTX / "scale_add_f32.ptx", symbol="scale_add_f32", block=(128,), grid=lambda Nn: (tilefish.cdiv(Nn, 128),), reference=scale_add_reference
)
def scale_add(x: Tensor[Nn, f32], y: Out[Tensor[Nn, f32]], n_elem: Nn, k: const[i32], c: const[i64]): ...


@tilefish.graph
def two_scales(x: Tensor[Rg, 64, f32]):
    """``(x * 3 - 2) * -1 + Rg``: one binary launched twice, with the values each call gives (the second ``c`` is the bucket's ``Rg``)."""
    a = scale_add(x.reshape(Rg * 64), k=3, c=-2)
    return scale_add(a, k=-1, c=Rg).reshape(Rg, 64)
