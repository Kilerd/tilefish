# Numerics: the CUDA math guard, the procedure on a CUDA or torch change, and decisions

The numerics contract of a lowered kernel is the docstring of `tilefish.lower` (its table and notes 1 to 20): every op, the PTX it becomes,
and what its bits are, measured. This document is about what that contract depends on outside tilefish, how to check it again when that
changes, and the decisions the contract rests on.

## 1. What the bits of a lowered kernel depend on

Most ops of a lowered kernel are one PTX instruction or a fixed sequence that tilefish writes (`div.rn.f32`, `fma.rn.f32`, `sin.approx.f32`, the
`add.rn` chains of an ordered sum, ...). Their bits depend on the PTX, which is in the generated source, and on the hardware; nothing outside
tilefish changes them. Two kinds of op are different:

* **Copies of a CUDA math library function.** In IEEE math (note 13 of `tilefish.lower`) `exp` and `silu` are built on CUDA's `expf`, `rsqrt` is
  `rsqrtf` and `erf` is `erff` (note 15): the inline PTX that NVRTC emits for each, copied once. The binary's bits do not move, but their claim, "the
  bits of CUDA C++'s `expf`" (and of torch's `exp`, built with it), holds only for the CUDA releases whose function is that sequence.
* **libdevice functions.** `sin`, `cos` and `tanh` (either math), `gelu` and the default math's `erf` call libdevice, which Triton links from a file
  (its own `libdevice.10.bc`, or the one `TRITON_LIBDEVICE_PATH` names). Another file gives other bits, and their claim (note 10: "torch's bits but on
  a subnormal") holds only for the releases whose `sinf`, `cosf`, `tanhf` are libdevice's.

* **Arithmetic that the PTX leaves to ptxas.** An op that is plain PTX (Triton's `a * b`, `a + b`) is assembled as ptxas decides: with
  `enable_fp_fusion=True` it may contract a multiply and an add into one rounding, which the contract allows (note 2). With
  `enable_fp_fusion=False` it must not, and ptxas 12.9 does anyway where LLVM has packed two lanes into `mul.rn.f32x2` / `add.rn.f32x2` (sm_100
  and later: `FFMA2` on sm_100 and sm_103, `FFMA` on sm_120). So a kernel with `fp_fusion=False` writes each float `+ - *` as one scalar `.rn`
  instruction of inline PTX (note 2), and the contraction check reads every binary built with `enable_fp_fusion=False` (section 4).

These functions do change between releases. Measured over every f32 on a B300 (sm_103), compiled by each release's NVRTC with nvcc's defaults:

| function | CUDA releases with the bits of tilefish's op | |
|---|---|---|
| `expf`, `rsqrtf` | 12.8, 12.9, 13.0, 13.1, 13.2 | the PTX is the same text in 13.5 |
| `sinf`, `cosf`, `tanhf` | 12.8, 12.9, 13.0, 13.1, 13.2 | `tanhf`'s PTX changed in 13.0 (`copysign`), not its bits; 13.5's `sinf` and `cosf` are unrolled differently (not measured: the driver of the test machine is 13.2) |
| `erff` | 12.8, 12.9, 13.0, 13.1 | 13.2 changed it (other coefficients and threshold): it differs on 353 987 744 inputs, by one ULP |

This table is `tilefish._cudamath.VERIFIED`. libdevice's own `erff` (`gelu`'s) is none of them: it differs from CUDA's on 24 071 628 inputs, and its
contract is a tolerance, not a release.

## 2. The guard

* **Recorded.** The manifest entry of a generated kernel that has such ops carries `cuda_math` (manifest schema 3, section 4.7): the functions, the
  releases of the table on which every one of them was measured, and the sha256 of the libdevice it was linked with. It is in the manifest digest. The
  build reads the functions from the code of the generated source (the helpers and calls `tilefish.lower` writes for them); the source is not
  changed, so no source digest moves.
* **Checked in Python.** A `jit`, `replay` or `reference` run of such a kernel (and the replay of a plan with one) compares the release torch is built with (`torch.version.cuda`)
  with the entry's releases, and a replay also compares the libdevice a build in the process would link with the entry's. A mismatch is a
  `tilefish.TilefishCudaMathWarning`: the binary runs as it was exported, and the warning says that torch's own ops, and the reference mode that uses
  them, may not be its bits here. To refuse instead: `warnings.simplefilter("error", tilefish.TilefishCudaMathWarning)` (or `pytest -W
  error::tilefish.TilefishCudaMathWarning`).
* **Checked in Rust.** The crate reads `Entry::cuda_math` and cannot tell the release of the host's other CUDA code, so a host says it:
  `manifest.check_cuda_math("13.0")?` refuses a manifest whose generated kernels' functions were not measured on that release, naming each.

What it does not cover: a hand-written `extern` that calls libdevice (its identity is its IR; its math is its author's), and a release nobody has
measured (it is simply not in the table, which the guard reports as a mismatch).

## 3. The procedure on a CUDA or torch change

Run it on a GPU whenever torch is upgraded to another CUDA build, a new CUDA release is used for the code tilefish's kernels are compared with,
Triton (and so its libdevice) changes, or on a new GPU architecture.

1. Get the NVRTC of each release to check: the `nvidia-cuda-nvrtc` wheel of that release (`pip download --no-deps nvidia-cuda-nvrtc==13.2.86`, or
   `nvidia-cuda-nvrtc-cu12==12.9.86` for CUDA 12), unpacked anywhere: `libnvrtc.so.13` is under `nvidia/cu13/lib` (`nvidia/cuda_nvrtc/lib` for 12). Its
   `libnvrtc-builtins` must be next to it. The driver must be of that release or later (it compiles the PTX).
2. Run the exhaustive comparisons:

   ```sh
   cd python
   TILEFISH_NVRTC_PATHS=/path/to/13.2/libnvrtc.so.13:/path/to/12.9/libnvrtc.so.12 pytest -m cuda_math -s
   ```

   `-m cuda_math` selects the comparisons with CUDA's math library over every f32: `test_gpu_cuda_math.py` (the table, for torch's NVRTC and each
   one named), `test_gpu_ieee.py`, `test_gpu_tanh.py`, `test_gpu_trig.py` (against torch's ops) and `test_gpu_primitives.py` (against NVRTC and PTX).
   About half a minute on a B300 MIG slice, plus a few seconds per extra release.
3. Read the result. `test_the_table_is_what_each_release_computes` fails when the table is out of date, and says how: a listed release that differs
   (take it out of `VERIFIED`: kernels with that function then warn on it) or an unlisted one that is the op's bits on every f32 (add it). The other
   files fail when torch's own ops are not the binaries' any more (a torch built with a release whose function changed): the warning of section 2
   is what a user sees then, and the contract notes that name the measured releases need the new one.
4. Update `tilefish._cudamath.VERIFIED` and the table of section 1, re-run step 2 until it passes, and export again the manifests whose `cuda_math`
   should list the new release (an old manifest keeps the list it was exported with).

## 4. Decisions

* **`num_warps` is part of every kernel's identity** (the declaration digest, the compile spec, the manifest entry and its digest; a replay
  refuses another). It is what Triton's own reductions depend on: `sum` without an order, `mean` and `max` reduce in a tree that the warp count
  shapes (on 1024 rows of values of every magnitude, 669 to 878 rows differ between 1 warp and 2 to 16, at 96, 1000 and 4096 columns). Per element
  ops, casts, `argmax` / `argmin` (a total order) and ordered sums (note 17) do not depend on it: their bytes are the same at every warp count
  (`tests/test_gpu_reduction_order.py`). One rule for every kernel is checked mechanically where a per-kernel argument would not be; a kernel whose
  every reduction has an `order` can change its warps for speed without changing its bytes. The contract's paragraph "num_warps and the numerics
  identity" in `tilefish.lower` says the same.
* **A reduction order is part of the call, not of the template.** `tilefish.sum(x, -1, order=...)` names the order a CUDA kernel adds in (a
  sequential chain, or lanes of chains and a fixed tree over the lanes, with `fma=True` for chains of `acc = fma(a, b, acc)`), lowered with data
  movement and `add.rn` so that neither Triton's layouts nor `num_warps` enter it. Without `order` a sum keeps Triton's tree, which is faster.
* **The primitives copy CUDA's spelling**, not an idealised one: `-x` is CUDA's negation (`neg.f32` gives the canonical NaN, where a flip of the
  sign bit would keep the payload), `sin_approx` is the hardware's `sin.approx.f32` with its subnormal flush, `erf` in IEEE math is the `erff` of
  CUDA 12.8 to 13.1. What a CUDA C++ kernel compiled with nvcc's defaults computes is what the DSL can say.
* **"No contraction" is written out and checked on the binary.** `@tilefish.kernel(fp_fusion=False)` promises that every float `+`, `-` and `*`
  rounds on its own, and an `extern` declared with `enable_fp_fusion=False` asks for the same. The compile option (`--fmad=false`) does not keep
  that promise on sm_100 and later (above), so it is kept twice:
  * **by the lowering**: a kernel with `fp_fusion=False` (and every fused region) writes each float `+ - *` as one scalar instruction of inline
    PTX with `.rn`, which nothing packs and ptxas does not contract, and every narrowing cast as inline `cvt.rn` (notes 1 and 2 of
    `tilefish.lower`);
  * **by the contraction check** (`tilefish._fpcontract`, run by every build): a binary built with `enable_fp_fusion=False` (a lowered kernel, a
    fused region, a generated op or a hand-written Triton or Gluon kernel) is disassembled, and so is its PTX assembled again with the same flags
    and every float multiply replaced by a bitwise `or` of its operands, which multiplies nothing. A binary with more fused multiply-add lanes
    (`FFMA`, `FFMA2`, `HFMA2`, `DFMA`, not the idioms that multiply or add nothing) than that is refused at build, with the kernel's name, the
    instructions and the fix (scalar `.rn` inline PTX where the rounding matters, or `enable_fp_fusion=True`). The comparison takes in what
    ptxas fuses on its own account (an explicit `fma`, the expansion of an IEEE division, square root or reciprocal) without listing it. The
    disassembler is the first `nvdisasm` that decodes the architecture (`$TILEFISH_NVDISASM_PATH`, Triton's, the one next to its `ptxas`, an
    `nvidia-cuda-nvdisasm` wheel, the `PATH`); Triton 3.5.1 bundles CUDA 12.8's, which does not decode sm_103, and there the PTX is assembled for
    `sm_100a`, the base of its family, and read (`pip install nvidia-cuda-nvdisasm` reads sm_103 itself). A binary with no float multiply or
    no float add is not disassembled. `tests/test_gpu_fpcontract.py` builds a Triton and a Gluon kernel that ptxas contracts (refused on
    sm_100, sm_103 and sm_120, built on sm_90), the same kernel with scalar `.rn` instructions (built everywhere) and every case of the lowering
    with `fp_fusion=False` on all four.
  A negation's NaN is written out the same way (note 16): `neg.f32` gives the canonical NaN, but ptxas folds it into a select as a negated
  operand, which flips the NaN's sign.
* **The default GEMM posture keeps its rule: a GEMM with a bias gets no split-K** unless its declaration opts in (`allow_bf16_reduction`, or the
  opt-in of its operands' type). The rule is what lets the default posture promise one rounding of `A @ B + bias` for every pin it makes, on every
  device and cuBLASLt version, without a measurement per algorithm. A split-K with the bias epilogue was measured to add the bias to a sum already
  rounded to the output type, even with f32 partial sums (cuBLASLt 12.8.4, RTX 5090, algorithm 21, 32 x 1024 x 3072: 8409 of 32768 elements past
  what one rounding and an f32 accumulation allow). Another algorithm keeps the one rounding (algorithm 66 with a split-K of 3, which `addmm` runs
  for 32 and 64 x 1024 x 3072 with cuBLASLt 13.1.0 on sm_103), but nothing in the heuristic's answer tells the two apart, and the double rounding
  is silent. So it costs the default posture `addmm`'s bytes on 2 of the 15 biased shapes of `test_gpu_gemm.py` under 13.1.0 (all 15 under
  12.8.4), and a caller who wants `torch.addmm`'s bytes declares `heuristic="torch"`, which matches it on every shape measured (12.8.4 and
  13.1.0). Relaxing the rule per algorithm would need an exhaustive check of each split-K algorithm with the bias epilogue on each device and
  cuBLASLt version; it is not planned.

## 5. Device-global state: designed, not implemented

A counter that a kernel bumps on the device (how many values a saturating store clamped, how many tokens were dropped) and the host reads after a
run cannot be written today: a lowered kernel has no atomics, and a plan has no buffer that outlives its own writes with a defined reset.

It fits the model under three restrictions, which keep a run's bytes a function of its inputs:

* **A declared state buffer.** A graph declares it as a parameter of a new role, `State[DevScalar[i64]]` (an integer scalar, or a small integer
  tensor), with `reset="run"`: the plan zeroes it at the start of every run, as its first step, and it is returned (a host reads it after the run,
  or binds it to its own buffer). Not a caller-provided output: its value at the start of a run is part of the contract (zero), not whatever the
  caller's memory held.
* **Order-free updates only.** A kernel updates it with an integer atomic of a commutative and associative operation (`tilefish.atomic_add`, and
  `max`, `min`, `or`), on wrap-around integers: the final value is the same whatever the order in which programs reach it, so `jit`, `replay`, a
  CUDA graph and `reference` (which sums the per-program contributions in any order) agree byte for byte. A float atomic is refused: its result
  depends on the order. In a lowered kernel a program adds its own contribution once (an integer `tl.sum` over the block, exact), so a counter
  costs one atomic per program.
* **One run at a time per buffer.** The state lives in the plan's arena, which runs of one plan already serialise (and concurrent streams already
  need an arena each), so two runs never update one counter.

It should wait for two pieces it needs, which other work owns: a memset step in the plan schema (an open item: plan schema 1 has none, and the reset
is one), and the atomic ops in the DSL with their lowering in both templates. With those, the state buffer is an additive field of the plan
(`role: "state"`, its reset) and of the manifest ABI (an `inout` device scalar that a kernel may update atomically), read by both hosts; until
then a hand-written `extern` that atomically updates an `InOut` tensor the caller zeroes is the way to count.
