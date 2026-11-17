"""The impl ``cublaslt.matmul``: a GEMM with a cuBLASLt epilogue, its verification per pin and its canary (RFC 0003, milestone E2).

``@tilefish.library(..., impl="cublaslt.matmul", epilogue=..., beta=...)`` (:mod:`tilefish._library`) is a GEMM whose epilogue the library runs on
the f32 accumulator, before the one rounding to the result's type: ``out = epilogue(acc [+ c] [+ bias])``, with ``epilogue`` one of ``"none"``,
``"bias"``, ``"relu"`` and ``"relu_bias"`` (cuBLASLt's ``DEFAULT``, ``BIAS``, ``RELU``, ``RELU_BIAS``) and ``c`` read when ``beta=1``. Its bytes are
the library's under the pin (contract (iii) of RFC 0003): the algorithm, the cuBLASLt version, the SM count and the epilogue.

**Verification per pin.** Where the epilogue's arithmetic can be written down, the bytes may equal contract (ii): ``ρ(f(acc))``, the formula
(:func:`tilefish._library.formula`) computed in f32 on the accumulator of the pin and rounded once, with ``c`` added before the bias (the library's
order, measured: X1). That is a property of the pin, not of the epilogue (X6: one algorithm of cuBLASLt 13.1.0 adds ``c`` and a bias in neither
order), so every pin is verified when it is made (:func:`verify`, at ``jit`` and at ``export``), and its entry records the result:

* the accumulator is the f32 output of the pin's own configuration re-encoded with an f32 ``D`` (``"f32-form"``: X1 found it observable this way,
  its rounding the 16-bit output bit for bit), or, when cuBLASLt does not take that configuration with an f32 ``D``, of the f32 GEMM the same posture
  pins (``"plain-f32-pin"``);
* the operands are two seeded sets: ``a ~ N(0, 1)``, ``b ~ N(0, 1/k)``, a bias ``~ N(0, 1/4)`` and ``c ~ N(0, 1)``, the bias and ``c`` with
  special values in their first columns (``+0``, ``-0``, ``+inf``, ``-inf``, NaN, the largest finite value of both signs, a subnormal), and rows of
  ``a`` that make special accumulators, which every epilogue sees: a NaN, a zero and an infinity in each set, and in the second set two rows that
  overflow to ``+inf`` and ``-inf``;
* the reference is computed by torch on the device in f32 (``torch.relu``, IEEE adds) and rounded by torch's cast, which on CUDA gives the bits of
  tilefish's own lowering of the same ops (the canonical NaN; ``tilefish.lower`` note 1);
* ``"measured-exact"`` when every element of both sets is equal bit for bit, ``"binary-defined"`` otherwise, with the count of elements that
  differ and the largest distance in units in the last place (``verification`` of the entry). A declaration with ``contract="measured-exact"``,
  as every rewrite of a plan's arithmetic into an epilogue is, refuses a pin that is not.

**The canary.** cuBLASLt exposes no build number (``tilefish._cublaslt.version``), so ``export`` records ``{"seed": s, "out_digest": d}``: the pin run
on operands made of the seed on the device (:mod:`tilefish._canary`: ``a`` tag 0, ``b`` tag 1, ``bias`` tag 2, ``c`` tag 3, dense in their layouts),
``out`` and the workspace zeroed, and the digest of ``out``. A replay runs it once per process and device before the pin is used, and refuses a
different digest; the crate's load does the same (``docs/manifest-schema-3.md``, section 9.5).
"""

from __future__ import annotations

import inspect
import threading
from typing import Any

from .errors import TilefishManifestError, TilefishRuntimeError

CANARY_SEED = 1  # what export records (a cuDNN convolution's canary has the same)
VERIFY_SEEDS = (0, 1)  # the operand sets of the verification
SPECIAL_COLUMNS = 8
_SIZES = {"bf16": 2, "f16": 2}


def algo_text(algo: str) -> str:
    """An algorithm as a message names it: its id and the first bytes of its pin (``id 66, 42000000...``)."""
    try:
        from . import _cublaslt

        return f"id {_cublaslt.algo_info(algo)['id']} ({algo[:16]}...)"
    except Exception:  # noqa: BLE001 - a message names what it can
        return f"{algo[:16]}..."


def verification_text(v: "dict[str, Any]") -> str:
    """A verification record as a message says it."""
    if v["differ"] == 0:
        return f"all {v['elements']} elements equal {v['formula']} rounded once"
    return f"{v['differ']} of {v['elements']} elements differ from {v['formula']} rounded once, by up to {v['max_ulp']} ULP"


def _torch() -> Any:
    from . import _cublaslt

    return _cublaslt._torch()


def _specials(torch: Any, dtype: Any) -> Any:
    f = torch.finfo(dtype)
    return torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan"), f.max, f.smallest_normal / 4, -f.max], dtype=torch.float32)


def operands(spec: Any, seed: int, device: Any) -> "dict[str, Any]":
    """The operands of verification set ``seed`` for ``spec``, on ``device``: ``a`` and ``b`` in their stored layouts, the bias and ``c`` when the
    spec has them (see the module docstring)."""
    torch = _torch()
    t = getattr(torch, {"bf16": "bfloat16", "f16": "float16"}[spec.dtype])
    m, n, k = spec.m, spec.n, spec.k
    g = torch.Generator(device=device).manual_seed(1_000_003 * seed + 7919)
    a = torch.randn(m, k, device=device, generator=g)
    # rows of special accumulators, which every epilogue sees (a ReLU without a bias or c too): a NaN, a zero, an infinity of each sign by the column,
    # and, in the second set, rows that overflow to +inf and -inf on most columns
    special = {0: float("nan"), 1: 0.0, 2: float("inf")} if seed % 2 == 0 else {3: float("nan"), 4: 0.0, 5: float("-inf")}
    for row, value in special.items():
        if row < m:
            if value == 0.0:
                a[row] = 0.0  # every product +-0: the accumulator is +0
            else:
                a[row, 0] = value  # one element: NaN, or +-inf times b[:, 0] (NaN where b is 0), and finite products after it
    if seed % 2 == 1 and m >= 3:
        a[1] = float(torch.finfo(t).max)
        a[2] = -float(torch.finfo(t).max)
    b = torch.randn(n, k, device=device, generator=g) / k**0.5
    out: "dict[str, Any]" = {
        "a": (a if spec.layout[1] == "N" else a.t()).to(t).contiguous(),
        "b": (b if spec.layout[0] == "T" else b.t()).to(t).contiguous(),
    }
    cols = min(SPECIAL_COLUMNS, n)
    if spec.bias:
        bias = torch.randn(n, device=device, generator=g) * 0.5
        bias[:cols] = _specials(torch, t)[:cols].to(device)
        out["bias"] = bias.to(t)
    if spec.beta:
        c = torch.randn(m, n, device=device, generator=g)
        first = (seed * SPECIAL_COLUMNS) % max(n - cols + 1, 1)  # the first set has them under the bias's specials, the second next to them
        c[:, first : first + cols] = _specials(torch, t)[:cols].to(device)
        out["c"] = c.to(t)
    return out


def reference(spec: Any, acc: Any, ops: "dict[str, Any]") -> Any:
    """The formula of the spec (:func:`tilefish._library.formula`) on the f32 accumulator ``acc``, in f32, rounded once to the result's type."""
    torch = _torch()
    x = acc
    if spec.beta:
        x = x + ops["c"].to(torch.float32)
    if spec.bias:
        x = x + ops["bias"].to(torch.float32)
    if spec.epilogue.startswith("relu"):
        x = torch.relu(x)
    return x.to(ops["a"].dtype)


def compare(torch: Any, got: Any, want: Any) -> "tuple[int, int]":
    """The elements of two 16-bit tensors whose bits differ, and the largest distance between them in units in the last place over the pairs that
    are both finite (in the order of the values: ``-0`` and ``+0`` are one apart)."""
    gb, wb = got.contiguous().view(torch.int16).to(torch.int32), want.contiguous().view(torch.int16).to(torch.int32)
    differ = int((gb != wb).sum())
    if differ == 0:
        return 0, 0

    def order(bits: Any) -> Any:
        mag = bits & 0x7FFF
        return torch.where(bits & 0x8000 != 0, -mag - 1, mag)

    finite = torch.isfinite(got.float()) & torch.isfinite(want.float())
    ulp = (order(gb) - order(wb)).abs()
    return differ, int(ulp[finite].max()) if bool(finite.any()) else 0


def _accumulator_algo(spec: Any, algo: str, index: int) -> "tuple[str, str]":
    """The algorithm whose f32 output is the accumulator of ``algo``'s problem, and where it comes from (:data:`tilefish._library.ACCUMULATORS`)."""
    from . import _cublaslt

    f32 = _cublaslt.Form(spec.dtype, "f32", spec.layout)
    common = {"workspace_bytes": spec.workspace_bytes, "device_index": index, "form": f32, "sm_count_target": spec.sm_count_target, **spec.opt_ins}
    try:
        same = _cublaslt.encode_algo(_cublaslt.algo_info(algo), device_index=index, form=f32)
        _cublaslt.check_algo(spec.m, spec.n, spec.k, False, same, **common)
        return same, "f32-form"
    except TilefishRuntimeError:
        pinned = _cublaslt.pin_algo(spec.m, spec.n, spec.k, False, heuristic=spec.heuristic, **common).algo_hex
        return pinned, "plain-f32-pin"


def verify(spec: Any, algo: str, index: int) -> "tuple[str, dict[str, Any]]":
    """Verify the pin ``algo`` of a ``cublaslt.matmul`` spec on device ``index`` against contract (ii) (the module docstring): its contract
    (``"measured-exact"`` or ``"binary-defined"``) and the record an entry keeps (``verification``)."""
    from . import _library

    torch = _torch()
    device = f"cuda:{index}"
    with torch.cuda.device(index):
        if torch.cuda.is_current_stream_capturing():
            raise TilefishRuntimeError(
                f"the cuBLASLt matmul '{spec.stem}' is pinned and its pin verified at its first jit run in the process, and this one is inside a CUDA "
                "graph capture: run it once before the capture"
            )
    acc_algo, source = _accumulator_algo(spec, algo, index)
    elements = differ = max_ulp = 0
    with torch.cuda.device(index):
        stream = torch.cuda.current_stream(index).cuda_stream
        for seed in VERIFY_SEEDS:
            ops = operands(spec, seed, device)
            out = torch.empty(spec.m, spec.n, dtype=ops["a"].dtype, device=device)
            acc = torch.empty(spec.m, spec.n, dtype=torch.float32, device=device)
            bias = ops["bias"].data_ptr() if spec.bias else None
            c = ops["c"].data_ptr() if spec.beta else None
            _library.run_gemm_ptrs(spec, algo, ops["a"].data_ptr(), ops["b"].data_ptr(), bias, out.data_ptr(), stream, index, c_ptr=c)
            _run_f32(spec, acc_algo, ops, acc, stream, index)
            want = reference(spec, acc, ops)
            d, u = compare(torch, out, want)
            elements += out.numel()
            differ += d
            max_ulp = max(max_ulp, u)
        torch.cuda.synchronize(index)
    record = {
        "formula": _library.formula(spec.epilogue, spec.beta),
        "accumulator": source,
        "operand_sets": len(VERIFY_SEEDS),
        "elements": elements,
        "differ": differ,
        "max_ulp": max_ulp,
    }
    return ("measured-exact" if differ == 0 else "binary-defined"), record


def _run_f32(spec: Any, algo: str, ops: "dict[str, Any]", acc: Any, stream: int, index: int) -> None:
    from . import _cublaslt

    f32 = _cublaslt.Form(spec.dtype, "f32", spec.layout)
    _cublaslt.matmul(
        ops["a"].data_ptr(), ops["b"].data_ptr(), acc.data_ptr(), None, spec.m, spec.n, spec.k, algo, workspace_bytes=spec.workspace_bytes,
        stream=stream, device_index=index, form=f32, sm_count_target=spec.sm_count_target, **spec.opt_ins,
    )  # fmt: skip


# -- the op a plan's rewrite calls -----------------------------------------------------------------------------------------------------------

_epilogue_ops: "dict[tuple[Any, ...], Any]" = {}
_epilogue_lock = threading.Lock()


def epilogue_op(gemm: Any, epilogue: str, beta: int) -> Any:
    """The ``cublaslt.matmul`` that ``tilefish.fusion(epilogues="library")`` calls instead of the GEMM ``gemm`` (a ``cublaslt.gemm`` or
    ``cublaslt.gemm_bias`` library op) and the arithmetic after it: named ``<gemm>__<epilogue>`` (``_c`` with ``beta=1``), the operands, layout,
    posture, workspace, opt-ins and SM count target of ``gemm``, a bias ``[N]`` and ``c [M, N]`` of the result's type (named ``bias`` and ``c``), its
    canary on and ``contract="measured-exact"``: a pin that is not equal to the formula is refused (the rewrite is exact against the graph's own
    arithmetic on the accumulator, which the formula is). The same GEMM and epilogue in two graphs is one operator."""
    from . import _library
    from .decorators import library
    from .dtypes import f32
    from .types import TensorType

    key = (gemm.name, _library.declaration_digest(gemm), epilogue, beta)
    with _epilogue_lock:
        found = _epilogue_ops.get(key)
    if found is not None:
        return found
    sig = gemm.signature
    out = sig.returns[0]
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    params = [inspect.Parameter("a", positional, annotation=sig.params[0].type), inspect.Parameter("b", positional, annotation=sig.params[1].type)]
    if epilogue in _library.MATMUL_WITH_BIAS:
        bias_t = sig.params[2].type if gemm.impl == _library.IMPL_GEMM_BIAS else TensorType((out.shape[1],), out.dtype)
        params.append(inspect.Parameter("bias", positional, annotation=bias_t))
    if beta:
        params.append(inspect.Parameter("c", positional, annotation=TensorType(tuple(out.shape), out.dtype)))
    name = f"{gemm.name}__{epilogue}{'_c' if beta else ''}"

    def fn(*args: Any) -> Any:  # a signature only: a library op has no body
        raise NotImplementedError

    fn.__signature__ = inspect.Signature(params, return_annotation=out)  # type: ignore[attr-defined]
    fn.__name__ = fn.__qualname__ = name
    fn.__module__ = "tilefish_generated"
    options = {
        flag: True for flag in ("allow_bf16_reduction", "allow_f16_reduction") if getattr(gemm, flag, False)
    }
    op = library(
        name, acc=f32, impl=_library.IMPL_MATMUL, epilogue=epilogue, beta=beta, heuristic=gemm.heuristic, workspace_bytes=gemm.workspace_bytes,
        layout=gemm.layout, sm_count_target=gemm.sm_count_target, contract="measured-exact", canary=True, **options,
    )(fn)  # fmt: skip
    with _epilogue_lock:
        return _epilogue_ops.setdefault(key, op)


# -- the canary ------------------------------------------------------------------------------------------------------------------------------


def canary_elements(spec: Any) -> "list[tuple[str, int, int]]":
    """The canary's operands of ``spec`` as ``(name, tag, elements)``, in the order of the entry's ABI: ``a`` (tag 0), ``b`` (1), the bias (2) and
    ``c`` (3) when the spec has them; each dense in its layout, of the operands' type."""
    out = [("a", 0, spec.m * spec.k), ("b", 1, spec.n * spec.k)]
    if spec.bias:
        out.append(("bias", 2, spec.n))
    if spec.beta:
        out.append(("c", 3, spec.m * spec.n))
    return out


def canary_digest(spec: Any, algo: str, seed: int, device_index: int) -> str:
    """The digest of the ``out`` the pin computes from the canary operands of ``seed`` (:func:`canary_elements`), on device ``device_index``.

    As a convolution's canary (``tilefish._conv.canary_digest``): everything on one stream made for it (non-blocking, ordered with no other), the
    operands made by the fill kernel, ``out`` (a whole number of words) and the workspace (the entry's ``workspace_bytes``) zeroed, the GEMM, the digest
    kernel and the copy of its 16 bytes back. Nothing of the operands or of ``out`` crosses to the host."""
    from . import _canary, _library

    torch = _torch()
    size = _SIZES[spec.dtype]
    device = f"cuda:{device_index}"
    with torch.cuda.device(device_index):
        stream = torch.cuda.Stream(device=device_index)
        with torch.cuda.stream(stream):
            made: "dict[str, Any]" = {}
            for name, tag, elements in canary_elements(spec):
                buf = torch.empty(max(elements * size, 16), dtype=torch.uint8, device=device)
                _canary.fill(buf.data_ptr(), elements, spec.dtype, seed, tag, stream.cuda_stream, device_index)
                made[name] = buf
            out = torch.zeros(-(-(spec.m * spec.n * size) // 4) * 4, dtype=torch.uint8, device=device)
            workspace = torch.zeros(spec.workspace_bytes, dtype=torch.uint8, device=device)
            _library.run_gemm_ptrs(
                spec, algo, made["a"].data_ptr(), made["b"].data_ptr(), made["bias"].data_ptr() if spec.bias else None, out.data_ptr(),
                stream.cuda_stream, device_index, workspace_ptr=workspace.data_ptr(), c_ptr=made["c"].data_ptr() if spec.beta else None,
            )  # fmt: skip
            sums = torch.zeros(2, dtype=torch.int64, device=device)
            _canary.digest(out.data_ptr(), out.numel() // 4, sums.data_ptr(), stream.cuda_stream, device_index)
            s1, s2 = (int(v) for v in sums.cpu())  # a copy on this stream, which it then waits for
    return _canary.hex_of(s1, s2)


_canary_lock = threading.Lock()
_canaries_passed: "set[tuple[Any, ...]]" = set()


def check_canary(built: Any, device_index: int) -> None:
    """Run the canary of a ``cublaslt.matmul`` pin read from a manifest (a ``BuiltLibrary``), once per process and device; refused:
    :class:`TilefishManifestError` (none recorded: nothing)."""
    canary = built.canary
    if canary is None:
        return
    spec = built.spec
    key = (device_index, spec.pinned_digest(built.sm_count), built.stem, built.arch, built.algo, built.cublaslt_version, canary["seed"], canary["out_digest"])
    with _canary_lock:
        if key in _canaries_passed:
            return
    torch = _torch()
    with torch.cuda.device(device_index):
        if torch.cuda.is_current_stream_capturing():
            raise TilefishRuntimeError(
                f"the cuBLASLt matmul '{built.stem}' ({built.arch}) runs its canary at its first replay in the process, and this one is inside a CUDA graph "
                "capture: run it once before the capture"
            )
    got = canary_digest(spec, built.algo, canary["seed"], device_index)
    if got != canary["out_digest"]:
        raise TilefishManifestError(
            f"the canary of the cuBLASLt matmul '{built.stem}' ({built.arch}, algorithm {algo_text(built.algo)}, epilogue {spec.epilogue}) gives out with "
            f"digest {got} here, and the manifest recorded {canary['out_digest']} (seed {canary['seed']}): this process's cuBLASLt (version "
            f"{built.cublaslt_version} too) or this device runs other kernels for the pin, so it is refused; export the manifest again with this cuBLASLt "
            "build, or run with the build that made it"
        )
    with _canary_lock:
        _canaries_passed.add(key)
