"""The ``reference`` mode: the traced kernel IR interpreted with torch.

It is an oracle for the Triton binaries, written to follow the numerics contract of :mod:`tilefish.lower`
node by node: a cast happens exactly where the IR has one (``.to(dtype)``, nowhere else), every other
operation runs in the dtype the IR gave its result, and a python scalar or ``const`` is a 0-d tensor of
that dtype (so a bf16 tensor combined with a float scalar rounds the scalar to bf16 first, as Triton does).
Reductions are torch's. ``/`` is IEEE division, ``exp``/``rsqrt``/``erf`` are torch's accurate versions,
where the binaries use ``div.full.f32``, ``ex2.approx`` and libdevice approximations; that is the
difference a tolerance has to cover, together with the reduction order and FMA contraction (see the
gpu tests for the measured numbers).

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


def _torch() -> Any:
    try:
        import torch
    except ImportError as e:
        raise TilefishRuntimeError("running a kernel needs torch: pip install 'tilefish[gpu]'") from e
    return torch


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

    def dtype_of(dt: Any) -> Any:
        return getattr(torch, TORCH_DTYPE_NAMES[dt.name].split(".")[1])

    def operand(x: "Value | Scalar") -> Any:
        if isinstance(x, Value):
            return env[x]
        return torch.tensor(x.value, dtype=dtype_of(x.dtype), device=device)  # the scalar, rounded to the IR dtype

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
                r = a[0].to(dtype_of(n.attrs["to"]))
            elif op in ("sum", "mean", "max"):
                axes = tuple(n.attrs["axes"])
                keep = bool(n.attrs["keepdims"])
                if op == "max":
                    r = torch.amax(a[0], dim=axes, keepdim=keep)
                else:
                    r = torch.sum(a[0], dim=axes, keepdim=keep)
                    if op == "mean":  # the contract: sum / D
                        count = 1
                        for ax in axes:
                            count *= a[0].shape[ax]
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
