"""Graphs whose bodies compute on the result of an extern with epilogue hooks (``docs/rfcs/0003-epilogue-fusion.md``, milestone E1): the patterns of
E1, each in four modes (``MODES``: fused and unfused, exact and accumulator), built over a set of externs by :func:`build`: the example externs
(``python/examples/epilogue_gemm.py``, :func:`examples`) on a GPU, and stand-ins with the same stubs and hooks (:func:`standins`, never compiled) on
any machine. For ``test_epilogue.py`` (no GPU), ``test_gpu_epilogue.py`` and the Rust and C++ gates (``rust_gate.py``: ``fused_epi_*``). :func:`mixed`
is a graph of E1 and E2 together (a cuBLASLt epilogue and a hooked extern on its accumulator).

No ``from __future__ import annotations``: tilefish evaluates annotations at declaration.
"""

import inspect
import pathlib
import sys

import tilefish
from tilefish import Dim, Out, Tensor, TensorDesc, bf16, const, f32

T = Dim("T", buckets=[64, 200])  # the rows of a GEMM (M)
H = Dim("H", 256)  # its K
F = Dim("F", 256)  # its N (and the gate and the up halves of a dual GEMM's 2F)
D = Dim("D", 128)  # the hidden size of a norm: a GEMM tile of 128 columns holds a whole row
Bc = Dim("Bc", buckets=[1, 3])  # a convolution's batch
Lc = Dim("Lc", buckets=[96, 500])  # its length
CI, CO, KW = 64, 128, 7
T128 = tilefish.tree(128, vector=8)

MODES = {
    "fused": tilefish.fusion(),
    "unfused": tilefish.fusion(grouping="none"),
    "acc": tilefish.fusion(epilogues="accumulator"),
    "acc_unfused": tilefish.fusion(epilogues="accumulator", grouping="none"),
}
EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"


def examples(arch: int):
    """The example externs for ``arch`` (``python/examples/epilogue_gemm.py``): ``(gemm, gemm_dual, conv)``; importing them imports Triton."""
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    import epilogue_gemm as eg

    # the GEMM's tiles are 128 columns wide on every architecture and its hooks see whole tiles (the norm's D: a tile holds a whole row)
    wide = {"BN": 128, "epilogue_subtiles": 1}
    return eg.gemm_for(arch, name="epi_gemm", **wide), eg.gemm_for(arch, name="epi_gemm_dual", split=2), eg.conv1d(CI, CO, KW, name="epi_conv")


def examples_subtiled(arch: int):
    """The example GEMM whose epilogue runs on two sub-tiles of the columns (its hooks called twice per tile), for the patterns that are elementwise."""
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    import epilogue_gemm as eg

    return eg.gemm_for(arch, name="epi_gemm_sub", BN=128, epilogue_subtiles=2)


def _stub(params, name):
    sig = inspect.Signature([inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=a, default=d) for n, a, d in params])

    def stub(*args, **kwargs): ...

    stub.__signature__ = sig
    stub.__name__ = stub.__qualname__ = name
    stub.__module__ = __name__
    return stub


_E = inspect.Parameter.empty
M, N, K = Dim("M"), Dim("N", divisible_by=16), Dim("K", divisible_by=16)
B, L = Dim("B"), Dim("L")


def _hooks(split=1):
    return (tilefish.hook("ACC", value="acc", out="c", split=split, block_n="BN"), tilefish.hook("EPILOGUE", value="out", out="c", split=split, block_n="BN"))


def standin_gemm(name="epi_gemm", split=1, BN=128):
    """A stand-in with the stub and the hooks of the example GEMM (``epilogue_gemm.gemm``), its kernel named by a string (never imported here)."""
    stub = _stub(
        [("a", TensorDesc(Tensor[M, K, bf16], block=(128, 64)), _E), ("b", TensorDesc(Tensor[N, K, bf16], block=(BN, 64)), _E), ("c", Out[Tensor[M, N, bf16]], _E),
         ("m", M, _E), ("n", N, _E), ("k", K, _E), ("BM", const, 128), ("BN", const, BN), ("BK", const, 64), ("SPLIT", const, split), ("GROUP_M", const, 8),
         ("EPI_SPLIT", const, 1)],
        name,
    )  # fmt: skip
    return tilefish.extern(triton="epilogue_gemm:gemm_kernel", name=name, grid=lambda M, N, BM, BN, SPLIT: (tilefish.cdiv(M, BM) * tilefish.cdiv(N, SPLIT * BN),),
                           epilogue=_hooks(split))(stub)  # fmt: skip


def standin_conv(name="epi_conv"):
    pad = (KW - 1) // 2
    stub = _stub(
        [("x", Tensor[B, L, CI, bf16], _E), ("w", Tensor[CO, KW * CI, bf16], _E), ("c", Out[Tensor[B, L, CO, bf16]], _E), ("nb", B, _E), ("l_in", L, _E),
         ("l_out", L, _E), ("CI", const, CI), ("CO", const, CO), ("KW", const, KW), ("PAD", const, pad), ("DIL", const, 1), ("BM", const, 128),
         ("BN", const, 64), ("BK", const, 64)],
        name,
    )  # fmt: skip
    return tilefish.extern(triton="epilogue_gemm:conv1d_kernel", name=name, grid=lambda B, L, BM, BN: (tilefish.cdiv(B * L, BM), tilefish.cdiv(CO, BN)),
                           epilogue=_hooks())(stub)  # fmt: skip


def standins():
    return standin_gemm(), standin_gemm("epi_gemm_dual", split=2), standin_conv()


# -- the patterns ---------------------------------------------------------------------------------------------------------------------------


def _gelu_tanh(h):
    return 0.5 * h * (1.0 + tilefish.tanh(0.7978845608 * (h + 0.044715 * h * h * h)))


def bodies(gemm, gemm_dual, conv):
    """``{name: (body, params)}`` of the patterns of E1 over these externs."""

    def bias_gelu(x, w, b):
        """GEMM, then a bias and erf GELU (contract (i): fused through EPILOGUE; (ii): R_acc rewrites y.to(f32))."""
        return tilefish.gelu(gemm(x, w).to(f32) + b.to(f32)).to(bf16)

    def bias_silu(x, w, b):
        return tilefish.silu(gemm(x, w).to(f32) + b.to(f32)).to(bf16)

    def bias_gelu_tanh(x, w, b):
        return _gelu_tanh(gemm(x, w).to(f32) + b.to(f32)).to(bf16)

    def periodic(x, w, b, a):
        """GEMM, a bias and a periodic activation with a per-channel a: h + sin(a h)^2 / a."""
        h = gemm(x, w).to(f32) + b.to(f32)
        s = tilefish.sin(a * h)
        return (h + s * s / a).to(bf16)

    def swiglu(x, w):
        """The dual-accumulator GEMM (gate and up halves of w) and silu(g) * u (Q9: two accumulators per program)."""
        g, u = gemm_dual(x, w).split(F, axis=-1)
        return (tilefish.silu(g.to(f32)) * u.to(f32)).to(bf16)

    def add_norm(x, w, res, nw):
        """GEMM (the output projection), a residual add and an rmsnorm; the residual stream a side output (Q10: whole-row tiles, D <= BN)."""
        h = gemm(x, w) + res
        h32 = h.to(f32)
        ms = tilefish.mean(h32 * h32, -1, keepdims=True, order=T128)
        return (h32 * tilefish.rsqrt(ms + 1e-6) * nw.to(f32)).to(bf16), h

    def add_norm_f32(x, w, res, nw):
        """The same with the residual stream in f32: y.to(f32) is widened, so R_acc reads the accumulator."""
        h = gemm(x, w).to(f32) + res
        ms = tilefish.mean(h * h, -1, keepdims=True, order=T128)
        return (h * tilefish.rsqrt(ms + 1e-6) * nw.to(f32)).to(bf16), h

    def conv_bias_silu(x, w, b):
        """A 1-D convolution (an implicit GEMM), a bias and SiLU."""
        return tilefish.silu(conv(x, w).to(f32) + b.to(f32)).to(bf16)

    def conv_periodic(x, w, b, a):
        h = conv(x, w).to(f32) + b.to(f32)
        s = tilefish.sin(a * h)
        return (h + s * s / a).to(bf16)

    gemm_in = [("x", Tensor[T, H, bf16]), ("w", Tensor[F, H, bf16])]
    conv_in = [("x", Tensor[Bc, Lc, CI, bf16]), ("w", Tensor[CO, KW * CI, bf16])]
    return {
        "bias_gelu": (bias_gelu, [*gemm_in, ("b", Tensor[F, bf16])]),
        "bias_silu": (bias_silu, [*gemm_in, ("b", Tensor[F, bf16])]),
        "bias_gelu_tanh": (bias_gelu_tanh, [*gemm_in, ("b", Tensor[F, bf16])]),
        "periodic": (periodic, [*gemm_in, ("b", Tensor[F, bf16]), ("a", Tensor[F, f32])]),
        "swiglu": (swiglu, [("x", Tensor[T, H, bf16]), ("w", Tensor[2 * F, H, bf16])]),
        "add_norm": (add_norm, [("x", Tensor[T, H, bf16]), ("w", Tensor[D, H, bf16]), ("res", Tensor[T, D, bf16]), ("nw", Tensor[D, bf16])]),
        "add_norm_f32": (add_norm_f32, [("x", Tensor[T, H, bf16]), ("w", Tensor[D, H, bf16]), ("res", Tensor[T, D, f32]), ("nw", Tensor[D, bf16])]),
        "conv_bias_silu": (conv_bias_silu, [*conv_in, ("b", Tensor[CO, bf16])]),
        "conv_periodic": (conv_periodic, [*conv_in, ("b", Tensor[CO, bf16]), ("a", Tensor[CO, f32])]),
    }


def graph_of(body, params, fusion, name):
    def fn(**kw):
        return body(*[kw[p] for p, _ in params])

    fn.__signature__ = inspect.Signature([inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=t) for p, t in params])
    fn.__name__ = name
    return tilefish.graph(fn, name=name, fusion=fusion)


def build(externs, prefix="epi"):
    """``{pattern: {mode: graph}}`` of every pattern over ``externs`` (``(gemm, gemm_dual, conv)``), each graph named ``<prefix>_<pattern>_<mode>``."""
    out = {}
    for name, (body, params) in bodies(*externs).items():
        out[name] = {mode: graph_of(body, params, fusion, f"{prefix}_{name}_{mode}") for mode, fusion in MODES.items()}
    return out


# -- E1 and E2 in one graph -----------------------------------------------------------------------------------------------------------------


def cublaslt_up(name):
    """A ``cublaslt.gemm`` of ``x [T, H]`` by ``w [F, H]`` (bf16, the default posture): the GEMM of :func:`mixed` that ``epilogues="library"`` rewrites."""
    sig = inspect.Signature(
        [inspect.Parameter("a", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Tensor[T, H, bf16]),
         inspect.Parameter("b", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Tensor[F, H, bf16])],
        return_annotation=Tensor[T, F, bf16],
    )  # fmt: skip

    def stub(*args): ...

    stub.__signature__ = sig
    stub.__name__ = stub.__qualname__ = name
    return tilefish.library(name, acc=f32, impl="cublaslt.gemm")(stub)


MIXED_PARAMS = [("x", Tensor[T, H, bf16]), ("w1", Tensor[F, H, bf16]), ("b1", Tensor[F, bf16]), ("w2", Tensor[F, F, bf16]), ("b2", Tensor[F, bf16])]


def mixed(gemm, up, name, mode="library", scope="accumulator", grouping="auto"):
    """E1 and E2 in one graph (``docs/rfcs/0003-epilogue-fusion.md``, "E1 and E2 together"): the cuBLASLt GEMM ``up`` (:func:`cublaslt_up`) and a
    bias, a cuBLASLt epilogue (``BIAS``) under ``mode``, then the extern ``gemm`` (with both hooks) and a bias + SiLU in a ``tilefish.fused(epilogues=
    scope)`` scope. With the defaults: one ``cublaslt.matmul`` step and one fused pair on the extern's accumulator; ``grouping="none"`` is its unfused
    composition (both rewrites kept: they are numerics, not fusion), which the fused plan must equal byte for byte."""

    def body(x, w1, b1, w2, b2):
        h = (up(x, w1).to(f32) + b1.to(f32)).to(bf16)
        with tilefish.fused(epilogues=scope):
            return tilefish.silu(gemm(h, w2).to(f32) + b2.to(f32)).to(bf16)

    return graph_of(body, MIXED_PARAMS, tilefish.fusion(epilogues=mode, grouping=grouping), name)
