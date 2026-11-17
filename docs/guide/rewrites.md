# Rewrites that know about rounding

Types make fusion legality mechanical: elementwise chains into GEMM
epilogues, a norm into the preceding residual add, quantization into the next
GEMM's prologue, permutations into strides instead of copy kernels, in-place
reuse of buffers. Bytes and FLOPs per operator follow from shapes and dtype
widths, so a build can report a roofline per kernel and count launches per
step.

Fusion can also change numerics. Every cast to a narrower type is a rounding
point in the IR, and rules come in two kinds:

- **exact**: every rounding point is kept and only kernel boundaries move.
  Applied automatically: the fusion of a graph body's arithmetic ([`fusion.md`](fusion.md)) is one.
- **numerics-changing**: a reduction order, an accumulator dtype or a rounding
  point changes. Applied only on request, and gated against the reference.

`num_warps` is part of a kernel's numerics identity in the same way: it fixes
the tree a `tl.sum` adds in, so the same kernel with another warp count differs
in the last bits. It is recorded in the declaration digest, the compile spec
and the manifest, and a replay refuses an entry that has another. Measured, it
is Triton's own reductions that depend on it; per-element ops, `argmax` and
sums with an `order=` have the same bytes at every warp count, so a kernel
whose every reduction has an order may change it for speed alone
(`docs/numerics.md`). So is FMA
contraction: a generated kernel always says whether `a * b + c` may be one
rounding (`@tilefish.kernel(fp_fusion=False)` says no), and a hand-written one
says it with `@tilefish.extern(triton=..., enable_fp_fusion=False)` (Triton's
compile option of that name; `None`, the default, leaves it to Triton). The
setting is in the declaration digest, every compile spec and the `options` of
the manifest entry, so a binary built with the other one is refused by a replay.
"No" is kept, not hoped for: the compile option alone does not hold on sm_100
and later, where LLVM packs two lanes' f32 operations and ptxas 12.9 contracts
the packed pair whatever `--fmad` says. So a generated kernel with
`fp_fusion=False` writes every float `+ - *` as one scalar `.rn` instruction,
and every binary built with `enable_fp_fusion=False`, a hand-written Triton or
Gluon kernel's included, is read back at build: one whose SASS has more fused
multiply-adds than the SASS of its PTX with the multiplies taken out is refused,
with the kernel's name and the fix (scalar `.rn` inline PTX for the roundings
that matter, or `enable_fp_fusion=True`; `docs/numerics.md`).

Whether a legal fusion is worth doing is still measured, not assumed. The
measurements live in the repository next to the rules.
