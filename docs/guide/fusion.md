# Fusion: arithmetic in a graph body

A graph body may compute. The arithmetic between two calls (`silu(g.to(f32)) * u.to(f32)` between two GEMMs, a bias and an activation, a residual add, a mask
from a device scalar, a dequantisation) is traced like a kernel body, with its rules (no implicit promotion, a scalar rounded to the tensor's dtype, f32 math
functions), and the planner groups it into **regions**: each a `@tilefish.kernel` of its own that tilefish generates (named `fused_<hash>` by its content, lowered
by the same templates), and one ordinary kernel step of the plan. Neither schema and neither host changes (`docs/rfcs/0001-plan-fusion.md`).

```python
@tilefish.graph(inputs=("x",))
def mlp(x: Tensor[M, H, bf16], w_gu: Tensor[2 * F, H, bf16], w_d: Tensor[H, F, bf16]):
    g, u = gemm_gu(x, w_gu).split(F, axis=-1)
    a = (tilefish.silu(g.to(f32)) * u.to(f32)).to(bf16)      # one kernel, not five
    return gemm_d(a, w_d)
```

**Fusion never changes bytes.** A fused plan writes the bytes of its *unfused composition*: every op a kernel of its own, every intermediate stored in the dtype the
IR gives it (`tilefish.fusion(grouping="none")` is that plan, and its taps show every intermediate). A region kernel contracts nothing (`fp_fusion=False`), and its code
does not depend on its neighbours (note 20 of `tilefish.lower`; the code of every kernel with `fp_fusion=False`, note 2): every float `+ - *` is one scalar `.rn`
instruction of inline PTX, because LLVM packs two lanes' f32 ops into `mul.rn.f32x2` / `add.rn.f32x2` and ptxas 12.9 contracts that pair (into `FFMA2` on sm_100 and
sm_103, `FFMA` on sm_120) even with `--fmad=false`; every narrowing float cast is inline PTX, because the folds of MLIR and LLVM keep an input NaN's payload through
`x.to(f32).to(bf16)` in one kernel; and a negation gives the canonical NaN explicitly, as in every kernel, because ptxas folds `neg.f32` into a select as a sign flip.
A reduction joins a region when its bytes cannot depend on the region: an ordered `sum` or `mean` (`order=`), the order-free `max`, `argmax`
and `argmin` (and `gather_rows`) are fused with the ops around them into **row regions**; a `sum` or `mean` without `order=` is a kernel of its own, its operand
stored (a Triton tree reduction adds in another tree when other loads share its kernel: 357 to 399 of 1024 rows differed, measured). An op whose value nothing reads
is not computed. One rounding of `a * b + c` is asked for with `tilefish.fma`.

```python
@tilefish.graph
def add_rmsnorm(x: Tensor[R, D, bf16], res: Tensor[R, D, bf16], w: Tensor[D, bf16]):
    h = x + res                                                           # the residual stream, returned too
    h32 = h.to(f32)
    ms = tilefish.mean(h32 * h32, -1, keepdims=True, order=tilefish.tree(128, offsets=(16, 8, 4, 2, 1, 64, 32), vector=8))
    return (h32 * tilefish.rsqrt(ms + 1e-6) * w.to(f32)).to(bf16), h      # one kernel, two outputs: the unfused composition is ten
```

A layer norm (two ordered means and an affine tail), softmax (a scale, a mask from a device scalar, `max`, `exp`, an ordered sum, a divide), a row-wise fp8
quantisation (an absolute maximum, a scale per row returned too, a saturating cast) and a norm over two column blocks of a GEMM's output (read where they are) are
each one kernel too. Row regions are lowered by revision 2 of the lowering (note 21 of `tilefish.lower`): an ordered sum takes its lanes' chains by halving
splits instead of a gather per step (the same adds in the same order, checked against CUDA kernels of each order at every length), a row of ordered sums has the
warps that suit it, and an fp8 cast is inline PTX. On a B300 slice the fused add + rmsnorm of 4096 rows of 1024 columns takes 16 us, its three kernels with a Triton
tree sum 65 us and its ten unfused kernels 103 us (the speed table is in `docs/rfcs/0001-plan-fusion.md`, phase 2).

**Checked at every export.** `export` runs every region of more than one op against its ops one by one, `jit` on the device, at every bucket, on seeded inputs that
reach every corner (NaNs with payloads, infinities, subnormals, signed zeros), and refuses the export at a difference (`verify=False` opts out; on a device of
another architecture the check is skipped with a `TilefishFusionWarning`). It writes `<graph>.fusion.json` next to the plan, the **fusion report**: for every
region the ops of the body it took, what it reads (where it is, or a copy), what it writes, every rounding it
keeps and the rewrites of the mode; `tilefish.fusion_report(graph)` and `tilefish.verify_fusion(graph)` give the same on demand. A view that is not
dense (the halves above, a head, a transpose) is read where it is, a stride per axis (by a row region of version 1, copied by `tilefish.contiguous`
first). The tests hold fused to unfused on every bf16 and f16 bit pattern, on grids of 2**22 bit patterns, on 120 random graphs of up to 24 ops and 80 of row ops,
on the row patterns above at every bucket (every bf16 pattern, rows of special values), and the inline casts to `.to()` on every f32.

**The mode** is declared, `@tilefish.graph(fusion=tilefish.fusion(...))`, and every choice is exact (the region's source carries it, so the pin does):

| parameter | values (default first) | what it says |
|---|---|---|
| `version` | `2`, `1` | the grouping rules, limits and rewrites, frozen: `2` fuses ordered reductions into row regions; `1` runs every reduction alone (the plans of tilefish 0.0.9 and 0.0.10, unchanged). A plan made with another version is refused, naming it |
| `grouping` | `"auto"`, `"none"` | regions as large as the rules allow (a region never spans a call, at most 64 ops), or one kernel per op |
| `math` | `"fast"`, `"ieee"` | the default math of the body's f32 ops; an op's own `math=` wins |
| `intermediates` | `"declared"`, `"opmath"` | every op rounds to its IR dtype; or torch's eager rule for a bf16 or f16 `+ - *` with a python scalar (computed in f32 with the scalar in f32, rounded once): with `math="ieee"` the bits are torch's eager composition (every non-NaN value, checked on every bit pattern) |
| `fp_contract` | `"none"`, `"mul-add"` | nothing contracted; or an f32 `a * b + c` (or `- c`) whose product nothing else reads becomes `tilefish.fma`, one rounding |
| `verify` | `True`, `False` | whether `export` checks the regions |

Each rewrite is made on the IR before the regions are formed, so its rounding points never depend on where the regions fall. `tilefish.barrier(x)` makes `x` a
boundary of the regions (it gets a buffer of the plan, which `taps=` reads); it never changes a byte. Not yet planned: a scatter in a graph body, a reduction without
`order=` joined to a region, a store into an `InOut` or carried tensor made by a region (it is a copy step), a dense reshape inside a region, a strided store,
`tilefish.fused(name=...)` scopes, `intermediates="f32"`, and modes that are not exact (`docs/rfcs/0001-plan-fusion.md`, section 11).
Fusion into a GEMM or a convolution, epilogues and prologues, is designed in RFC 0003 (`docs/rfcs/0003-epilogue-fusion.md`: the three numerics
contracts it keeps apart, the routes, and the experiments behind them); its first milestones, epilogue hooks on externs (E1) and cuBLASLt epilogues
(E2), are below. Nothing changes bytes without a declaration: `epilogues="exact"` is the default.

## Epilogue hooks: the arithmetic after an extern's GEMM, in its kernel

An extern says where its kernel takes an epilogue, and a graph that calls it and computes on its result gets one kernel instead of two (RFC 0003,
milestone E1). The kernel has a `tl.constexpr` hook parameter and its arguments, which the stub does not list, and calls the hook on each output tile
where it would store it (`tilefish._epilogue.ABI`):

```python
@triton.jit
def gemm_kernel(a, b, c, m, n, k, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ..., ACC: tl.constexpr, ACC_ARGS, EPILOGUE: tl.constexpr, EPILOGUE_ARGS):
    ...                                                                   # the main loop: acc, an f32 [BM, BN] tile
    tile = ACC(acc, rows, cols, rows < m, cols < n, ACC_ARGS)            # value="acc": the accumulator
    tile = EPILOGUE(tile.to(c.dtype.element_ty), rows, cols, rows < m, cols < n, EPILOGUE_ARGS)   # value="out": the rounded tile
    tl.store(c + rows[:, None] * n + cols[None, :], tile, mask=...)

@tilefish.extern(triton=gemm_kernel, grid=..., epilogue=(tilefish.hook("ACC", value="acc", out="c"), tilefish.hook("EPILOGUE", value="out", out="c")))
def gemm(a: TensorDesc(Tensor[M, K, bf16], block=(128, 64)), b: TensorDesc(Tensor[N, K, bf16], block=(128, 64)), c: Out[Tensor[M, N, bf16]], m: M, n: N, k: K, ...): ...

@tilefish.graph
def up(x: Tensor[T, H, bf16], w: Tensor[F, H, bf16], bias: Tensor[F, bf16]):
    return tilefish.gelu(gemm(x, w).to(f32) + bias.to(f32)).to(bf16)     # one step: the GEMM with the bias and the GELU in its hook
```

* **Exact by default** (`tilefish.fusion()`, `epilogues="exact"`): the region goes into the `value="out"` hook, which sees the tile the kernel rounded to
  its output type, as it would store it, so the fused op writes the bytes of the extern followed by the region (contract (i)), and `export` checks it
  at every bucket on seeded inputs with special values (`verify_fusion` too).
* **On the accumulator**, declared: `tilefish.fusion(epilogues="accumulator")`, or a scope, `with tilefish.fused(epilogues="accumulator"):` (never per
  call). The graph is then the rewrite `R_acc` of its body: every widening `y.to(f32)` of the extern's result is the extern's f32 result (the
  accumulator form of the extern writes it, `<extern>__acc`), one rounding fewer, and the region goes into the `value="acc"` hook. The bytes are
  those of the unfused plan of the rewritten program (contract (ii)), checked at export; they are not the extern-then-region bytes (about a third of a
  bias + activation's outputs differ), and they are at least as accurate.
* **What a hook takes**: an elementwise region over the result's shape (a bias, an activation, a periodic activation with a per-channel parameter, a
  residual add with the stream as a side output, a mask, a cast), the dual-accumulator tiles of a gated activation (`tilefish.hook(..., split=2)`: a
  program holds the tile of the gate and the matching tile of the up projection), and a row region (an rmsnorm) where a tile of the extern holds whole
  rows (`block_n=`, a Triton kernel). What it does not take stays route (a), a region after the call, and the fusion report, `boundaries()` and the
  advisor say why (a barrier, a view the hook does not hold, a call in between, a row wider than a tile, ...).
* **Speed is the kernel's**: the hook's arithmetic runs on the extern's warps after its main loop. On a B300 slice, with the example GEMM, a bias +
  GELU after a 2048 x 4096 x 1024 GEMM is 1.21 times faster fused than the GEMM followed by its region, a gated activation 1.14 times; an epilogue
  heavy for the kernel's warps can be slower (a `sin` per element after a convolution: 0.77 times; the table is in RFC 0003). `tilefish.barrier(y)`
  keeps the region after the call. `python/examples/persistent_gemm.py` is an example kernel (not a supported feature, `docs/scope.md`) that shows
  the hooks in a persistent, warp-specialised GEMM: Gluon on sm_100/103 (a TMA producer, a `tcgen05` issuer, its epilogue on partitions of its own
  that overlap the next tile's main loop), Triton on sm_120; what it measured is RFC 0003's experiment X8.
* **Nothing new for a host or a schema**: an extern with hooks is launched through a small generated module that calls its kernel with the identity
  hook (in its declaration digest), and a fused pair is a generated entry of its own, `<extern>_epi_<hash>`, its identity the IR it compiles to.
  `python/examples/epilogue_gemm.py` has a Triton GEMM (`tcgen05.mma` on sm_100/103, `mma.sync` on sm_120, both asserted in the SASS), its
  dual-accumulator form, a Gluon `tcgen05` GEMM and a 1-D convolution as an implicit GEMM, each with both hooks.

## cuBLASLt epilogues: the arithmetic after a cuBLASLt GEMM, in the library

`tilefish.fusion(epilogues="library")` (RFC 0003, milestone E2) declares that the arithmetic after a `cublaslt.gemm` (or `gemm_bias`) of a 16-bit result may run as one of its epilogues, on the accumulator:

```python
@tilefish.graph(fusion=tilefish.fusion(epilogues="library"))
def block(x: Tensor[M, H, bf16], w: Tensor[F, H, bf16], bias: Tensor[F, bf16], r: Tensor[M, F, bf16]):
    h = (proj(x, w).to(f32) + r.to(f32)) + bias.to(f32)                # c before the bias: the library's order
    return tilefish.where(h <= 0.0, 0.0, h).to(bf16)                   # ReLU with torch.relu's bits: one step, cublaslt.matmul
```

The pattern is `y.to(f32)`, then `+ c.to(f32)` (`[M, N]`), then `+ bias.to(f32)` (`[N]`), then `tilefish.where(h <= 0.0, 0.0, h)`, then `.to()` the GEMM's
type, each step optional but one, every value read by the next step alone and `c` and the bias made before the GEMM. It becomes one step of a
`cublaslt.matmul` declared `contract="measured-exact"`: the rounding of `y` is removed (contract (ii)), and every pin is verified equal to the region's
arithmetic on the accumulator, or the build is refused (`tilefish.barrier(y)` keeps the GEMM unfused). The pin is chosen for the epilogue, so its
accumulator is the GEMM's where the two pins compute the same sum (X1 found them the same on every shape but where 13.1.0 splits `k`); then a ReLU
alone writes the unfused plan's bytes (it commutes with the rounding), as the tests measured, but nothing checks that per pin. Another order (`(y + bias) + c` rounds otherwise in f32), an f32 bias or a value read elsewhere
is left as it is, and the fusion report (`library_epilogues`) says why; in an exact graph it lists what the mode would do, and the advisor reports such a
region as an action (`epilogue-library-cublaslt`). Measured on a B300 MIG slice (sm_103, cuBLASLt 12.8.4, CUDA graphs of 20 replays), the exact plan
(the pinned GEMM and a region) against the rewritten one, in microseconds:

| shape `M x N x K` | bias | bias + ReLU | residual | residual + bias |
|---|---|---|---|---|
| 640 x 4096 x 1024, bf16 | 24.8 → 20.6 (-17%) | 24.8 → 22.2 (-10%) | 24.9 → 22.6 (-9%) | 26.8 → 24.7 (-8%) |
| 640 x 1024 x 4096, bf16 | 18.5 → 16.5 (-11%) | 18.5 → 16.6 (-11%) | 18.6 → 16.6 (-11%) | 18.6 → 18.5 (-0%) |
| 2048 x 3072 x 1024, bf16 | 54.5 → 41.1 (-25%) | 55.3 → 43.1 (-22%) | 61.8 → 46.5 (-25%) | 65.8 → 49.3 (-25%) |
| 128 x 3072 x 1024, bf16 | 8.3 → 6.3 (-24%) | 8.3 → 6.3 (-24%) | 8.3 → 6.3 (-24%) | 8.3 → 6.3 (-24%) |
| 8192 x 3072 x 768, f16 | 192.7 → 124.7 (-35%) | 192.5 → 135.7 (-30%) | 208.0 → 136.3 (-34%) | 215.9 → 147.0 (-32%) |

The bytes differ from the exact plan's on 14 to 29% of the elements (one rounding fewer). On sm_120 (an RTX 5090) cuBLASLt's one algorithm for an
epilogue maps a ReLU's NaN to 0, so a ReLU (and some bias + residual shapes) is refused there under `"library"`, never run with other bytes (RFC
0003, 7.3). GELU (binary-defined), cuDNN fused graphs and generated GEMMs come later.

**One graph, several modes.** `epilogues=` is `"exact"` (the default), `"accumulator"` or `"library"`, per graph or per `tilefish.fused` scope (the
innermost wins for the ops traced in it; never per call). `"library"` keeps the exact hooks of externs and adds the cuBLASLt rewrite; `"accumulator"`
applies `R_acc` to the externs with a `value="acc"` hook and leaves a cuBLASLt GEMM as it is (`R_acc` of a library call is not planned: the GEMM keeps
its rounding, and the fusion report's `library_epilogues` says so). A cuBLASLt pattern is rewritten only where all its ops are in a `"library"` mode. A
graph that wants both declares one and scopes the other:

```python
@tilefish.graph(fusion=tilefish.fusion(epilogues="library"))            # the cuBLASLt GEMM's bias: one cublaslt.matmul
def layer(x: Tensor[M, H, bf16], w: Tensor[F, H, bf16], bias: Tensor[F, bf16], w2: Tensor[N, F, bf16], b2: Tensor[N, bf16]):
    h = (proj(x, w).to(f32) + bias.to(f32)).to(bf16)
    with tilefish.fused(epilogues="accumulator"):                         # the extern's GELU on its accumulator (contract (ii))
        return tilefish.gelu(gemm(h, w2).to(f32) + b2.to(f32)).to(bf16)
```
