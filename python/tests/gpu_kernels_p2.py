"""A hand-written attention kernel for the gpu tests of launch plans (not collected: the name has no ``test_`` prefix).

Written from scratch for these tests, and public. Importing this module imports triton, so only the gpu tests do: ``plan_model`` declares the extern
by the string ``"gpu_kernels_p2:attention_kernel"``, which the build imports. No ``from __future__ import annotations``: tilefish evaluates annotations.

One program per (query row, head): the row's query against the keys ``j < n_valid`` (a device scalar: one binary serves every ``n``), softmax in f32, the
weighted sum of the values. ``q``, ``k`` and ``v`` are *views* of one fused qkv tensor: only their row and head strides are passed, the last axis has stride 1.
"""

import triton
import triton.language as tl


@triton.jit
def attention_kernel(q, k, v, o, n_valid, q_s0, q_s1, k_s0, k_s1, v_s0, v_s1, rows, DH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    heads = tl.num_programs(1)
    d = tl.arange(0, DH)
    j = tl.arange(0, BLOCK)
    n = tl.load(n_valid)
    keep = (j < n) & (j < rows)
    qv = tl.load(q + row * q_s0 + head * q_s1 + d).to(tl.float32)
    kv = tl.load(k + j[:, None] * k_s0 + head * k_s1 + d[None, :], mask=keep[:, None], other=0.0).to(tl.float32)
    s = tl.sum(kv * qv[None, :], axis=1) * 0.125  # 1 / sqrt(64)
    s = tl.where(keep, s, float("-inf"))
    p = tl.exp(s - tl.max(s, axis=0))
    p = p / tl.sum(p, axis=0)
    vv = tl.load(v + j[:, None] * v_s0 + head * v_s1 + d[None, :], mask=keep[:, None], other=0.0).to(tl.float32)
    out = tl.sum(p[:, None] * vv, axis=0)
    tl.store(o + (row * heads + head) * DH + d, out.to(tl.bfloat16))
