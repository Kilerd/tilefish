"""The ``reference`` mode: the traced kernel IR interpreted with torch.

It is an oracle for the Triton binaries, written to follow the numerics contract of :mod:`tilefish.lower`
node by node:

* a cast happens exactly where the IR has one (``.to(dtype)``, nowhere else); a cast to ``e4m3`` or ``e5m2``
  is ``satfinite`` like the binaries (saturating at 448 and 57344, where torch gives NaN or infinity), and the
  half types round to nearest even like torch. A float to integer cast is torch's, which is a plain truncation
  in range; the lowering rejects it (rule ``cast.float-to-int``), so no binary has a behavior to compare with;
* every other operation runs in the dtype the IR gave its result, and a python scalar or ``const`` is a 0-d
  tensor of that dtype (a bf16 tensor combined with a float scalar rounds the scalar to bf16 first, as
  Triton does);
* ``x / c`` with a scalar or ``const`` divisor is ``x * f32(1 / c)``, which is what the binary computes
  (``div.full.f32`` of a constant divisor is a multiply by the correctly rounded f32 reciprocal), and so
  is ``mean`` (``sum / D``);
* reductions are torch's.

What stays different, and what a tolerance has to cover, is everything the binary does approximately:
``/`` between tensors (``div.full.f32``), ``exp`` (``ex2.approx``), ``rsqrt``, ``erf`` (libdevice), the
order of a reduction, and FMA contraction (the binaries fuse ``a * b + c`` into one rounding, the
reference rounds twice). The gpu tests measure the resulting error per kernel.

An ``InOut`` or ``Out`` parameter is written once, with the final version stored into it, after the whole
body has been evaluated, like the lowered kernel does: reads never see a store.
"""

from __future__ import annotations

from typing import Any

from ._launch import TORCH_DTYPE_NAMES
from .decorators import KernelOp
from .errors import TilefishRuntimeError
from .ir import Scalar
from .trace import Value

_E4M3_MAX = 448.0
_E5M2_MAX = 57344.0


def _torch() -> Any:
    try:
        import torch
    except ImportError as e:
        raise TilefishRuntimeError("running a kernel needs torch: pip install 'tilefish[gpu]'") from e
    return torch


def torch_dtype(torch: Any, dt: Any) -> Any:
    return getattr(torch, TORCH_DTYPE_NAMES[dt.name].split(".")[1])


def reciprocal(torch: Any, value: float, dtype: Any, device: Any) -> Any:
    """``1 / value`` correctly rounded in ``dtype``, as a 0-d tensor (computed on the CPU, where division is IEEE)."""
    one = torch.ones((), dtype=dtype)
    return (one / torch.tensor(value, dtype=dtype)).to(device)


def cast(torch: Any, x: Any, src: Any, dst: Any) -> Any:
    """``x.to(dst)``, with ``satfinite`` for the fp8 types.

    ``src`` and ``dst`` are tilefish dtypes. Float to ``e4m3`` / ``e5m2``: round to nearest even, saturating at
    the largest finite value (448, 57344), infinities included, instead of overflowing to NaN or infinity as
    torch does; NaN stays NaN. Every other cast is torch's (a float to integer cast is not lowered, see
    ``cast.float-to-int``, so only its in-range truncation toward zero is meaningful here).
    """
    target = torch_dtype(torch, dst)
    if src.is_float and dst.name == "e4m3":
        return x.to(torch.float32).clamp(-_E4M3_MAX, _E4M3_MAX).to(target)
    if src.is_float and dst.name == "e5m2":
        return x.to(torch.float32).clamp(-_E5M2_MAX, _E5M2_MAX).to(target)
    return x.to(target)


def interpret(kernel: KernelOp, tensors: "dict[str, Any]") -> "list[Any]":
    """Evaluate the kernel body on ``tensors`` (name -> tensor, every parameter; ``Out`` ones are filled in place).

    Returns the returned values as fresh tensors, in order. ``InOut`` and ``Out`` tensors are written in place.
    """
    torch = _torch()
    graph = kernel.graph
    device = next(iter(tensors.values())).device
    env: dict[Value, Any] = {}
    for p in graph.inputs:
        if p.access != "out":
            env[p] = tensors[p.name]

    def operand(x: "Value | Scalar") -> Any:
        if isinstance(x, Value):
            return env[x]
        return torch.tensor(x.value, dtype=torch_dtype(torch, x.dtype), device=device)  # the scalar, rounded to the IR dtype

    final: dict[str, Any] = {}
    with torch.no_grad():
        for n in graph.nodes:
            op = n.op
            if op == "store":
                value = env[n.inputs[1]]  # type: ignore[index]
                if n.outputs:  # an InOut parameter: reading it from here on reads the stored value
                    env[n.outputs[0]] = value
                if n.inputs[0] is not n.inputs[1]:  # `x[...] = x` writes nothing
                    final[n.attrs["target"]] = value
                continue
            if "kind" in n.attrs:
                raise TilefishRuntimeError(
                    f"reference mode does not interpret the call to the {n.attrs['kind']} op '{op}' inside kernel "
                    f"'{kernel.name}'; only the built-in ops are interpreted"
                )
            a = [operand(i) for i in n.inputs]
            (out,) = n.outputs
            if op == "add":
                r = a[0] + a[1]
            elif op == "sub":
                r = a[0] - a[1]
            elif op == "mul":
                r = a[0] * a[1]
            elif op == "div":
                divisor = n.inputs[1]
                if isinstance(divisor, Scalar) and divisor.dtype.name == "f32":  # a constant divisor: a multiply by 1/c
                    r = a[0] * reciprocal(torch, float(divisor.value), torch.float32, device)
                else:
                    r = a[0] / a[1]
            elif op == "rsqrt":
                r = torch.rsqrt(a[0])
            elif op == "exp":
                r = torch.exp(a[0])
            elif op == "sqrt":
                r = torch.sqrt(a[0])
            elif op == "silu":  # x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
                r = a[0] * (1.0 / (1.0 + torch.exp(-a[0])))
            elif op == "gelu":  # exact erf form, evaluated left to right
                r = 0.5 * a[0] * (1.0 + torch.erf(a[0] * 0.7071067811865476))
            elif op == "cast":
                src = n.inputs[0].type.dtype  # type: ignore[union-attr]
                r = cast(torch, a[0], src, n.attrs["to"])
            elif op in ("sum", "mean", "max"):
                axes = tuple(n.attrs["axes"])
                keep = bool(n.attrs["keepdims"])
                if op == "max":
                    r = torch.amax(a[0], dim=axes, keepdim=keep)
                else:
                    r = torch.sum(a[0], dim=axes, keepdim=keep)
                    if op == "mean":  # the contract: sum / D, with D a constant
                        count = 1
                        for ax in axes:
                            count *= a[0].shape[ax]
                        if r.dtype == torch.float32:
                            r = r * reciprocal(torch, float(count), torch.float32, device)
                        else:
                            r = r / count
            else:
                raise TilefishRuntimeError(f"reference mode does not interpret the op '{op}' (kernel '{kernel.name}')")
            env[out] = r
        results = []
        for v in graph.outputs:
            r = env[v]
            results.append(r.clone() if (v.is_param or any(r is t for t in tensors.values())) else r)
        for name, value in final.items():
            tensors[name].copy_(value)
    return results
