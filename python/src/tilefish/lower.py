"""Lowering of ``@tilefish.kernel`` bodies to Triton source (M2a).

``lower(kernel_op)`` turns the typed SSA IR of a kernel into one ``@triton.jit`` function in a
complete Python module, and describes that function as an :class:`~tilefish.decorators.ExternOp`
so :func:`tilefish.aot.compile_specs` yields the ahead-of-time compile inputs for it unchanged.
Nothing here imports triton or compiles anything. The output is deterministic: the same IR
gives the same bytes.

Two templates exist; a kernel that fits neither raises :class:`~tilefish.errors.TilefishLoweringError`
naming the rule that failed (``rule`` attribute). Without ``template=`` the elementwise template
is tried first, then the row template.

**row** (``R`` rows, one program per row, grid ``(R,)``). Every tensor is ``[R, D]``, ``[D]``
(shared by all rows), ``[R]`` or ``[R, 1]`` (one value per row), except a table of ``gather_rows`` or
``scatter_rows`` (``[N, D]`` or ``[N]``, addressed by an index; note 9). A static slice ``x[k]`` of a read-only
parameter is one of these at a static offset: ``codes[k]`` of a ``[K, R]`` codes plane is an ``[R]`` operand (the index of a
gather, say), ``tables[k]`` of a ``[K, N, D]`` parameter a table (note 19). Reductions (``sum``, ``mean``,
``max``, ``argmax``, ``argmin``) run over the last axis of an ``[R, D]`` tensor only. ``D`` is static, or bucketed (one
compile spec per bucket); ``R`` is dynamic and only appears in the grid. Prologue::

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    offs = row * D + cols

``D`` and ``BLOCK_D = next_pow2(D)`` are ``tl.constexpr`` parameters. ``[R, D]`` tensors are
addressed with ``offs``, ``[D]`` with ``cols``, per-row tensors with ``row``. ``[D]`` tensors are
read-only (storing one from every row program would race). ``num_warps`` is one warp per 256
lanes of ``BLOCK_D``: ``clamp(BLOCK_D // 256, 1, 16)``, and each bucket of a bucketed ``D`` gets the
warps of its own ``BLOCK_D`` (the spec of the ``D = 96`` binary says 1 warp whether or not a
``D = 4096`` bucket is declared, so the bits of a binary never depend on the other buckets; note 6).
``@tilefish.kernel(num_warps=N)`` or ``lower(num_warps=N)`` pins ``N`` for every bucket.

**elementwise** (grid ``(cdiv(n, 1024),)``). Every tensor has the same shape, flattened to
``n = prod(shape)`` (but an ``[N]`` table of ``gather_rows`` / ``scatter_rows``, addressed by an index),
or the trailing axes of it: such an operand (``pe: [T, D]`` of a ``[K, T, D]`` kernel) broadcasts over the
leading axes, lane ``o`` reading its element ``o % pe_n`` (``pe_n`` a runtime ``i32``, its element count), and
is never written. Any other shape that broadcasts to the kernel's (numpy's rules: ``[H, 1, L]`` or ``[H, L, 1]``
of an ``[H, L, L]`` kernel) is read at the offset of the lane's coordinates, and an ``iota`` along any axis is
the lane's coordinate there (note 12). Python and ``const`` scalars broadcast too; there are no reductions. ``n`` is a
runtime ``i32`` parameter whose dimension is that product (``3072*M``), so a spec says how to
compute it and ``compile_specs`` emits ``tt.divisibility`` when the product is provably a multiple
of 16. When the product is not affine (two symbolic dimensions), ``n`` gets a fresh dimension that
carries the largest power of two (up to 1024) the factors prove between them.
``BLOCK: tl.constexpr = 1024`` and ``num_warps = 4``. Prologue::

    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

A ``DevScalar[dtype]`` parameter (one element in device memory) is a plain pointer parameter that the kernel reads with ``tilefish.load(s)``, which is
``tl.load(s)``: no offsets, no mask, one load whose scalar broadcasts against the tensors it meets (the result has rank 0, and so does what
is computed from it alone). It does not take part in the shape of the kernel (the elementwise template's ``n`` is the element count of a tensor
parameter, never of a scalar); a scalar cannot be returned. A view (``reshape``, ``permute``, ``select`` and ``split`` are zero-copy views) is not
lowered: rule ``view`` or ``split`` says so, and a strided view goes to an ``extern`` with a ``Strided[...]`` parameter. The one exception is a
static slice in the row template, ``x[k]`` (``x.select(0, k)``) of a read-only tensor parameter, read at its offset (note 19).

Both templates: offsets are ``i32``, so a tensor may have at most ``2**31 - 1`` elements. A tensor
that provably has more is rejected (rule ``index-range``; the largest bucket of a bucketed
dimension counts, and every dimension is at least 1). A tensor whose size has no static bound
(a dynamic dimension) gets a ``max_elements`` precondition in every spec, holding the argument,
its shape and the bound, which a host checks at launch; a tensor that is statically small gets
none. The generated code is the same either way. No loops, no strides (a View, a ``split`` or a
call to a declared operator is rejected), ``num_stages = 3`` in the spec (irrelevant without loops).

Compile options in every spec of a generated kernel: ``num_warps`` and ``num_stages``, and
``enable_fp_fusion``, always written out: ``True``, which is Triton's default (the machine code is
byte-identical to a spec without the key), or ``False`` for ``@tilefish.kernel(fp_fusion=False)`` /
``lower(fp_fusion=False)``, whose every float ``+ - *`` rounds on its own (note 2: written out, and checked on the binary). The key is part of the
declaration digest and of what ``verify_against`` checks; a hand-written ``extern`` never has it.

Parameters of the generated function, in this order: the kernel's own parameters under their own
names (tensors are pointers, ``const`` parameters are ``tl.constexpr`` with the declared default
as the spec's constant, runtime dimension parameters are ``i32``), one output pointer per returned
value (``out``, or ``out0``, ``out1``, ... for several), then the template's parameters. A
generated name that collides with a kernel parameter is suffixed (``mask_1``). Constexpr values come
from the spec (``compile_specs(lowered.extern)``), not from defaults in the source.

Every pointer of a generated kernel is compiled as 16-byte aligned (``tt.divisibility=16``). The specialisation of a kernel for read-only
parameters aligned to their element only (``kernel.unaligned("a")``, which a graph calls for a dense view that does not provably start on 16 bytes)
is lowered to the same source: only its ``ExternOp`` differs, named ``<kernel>__ua<indices>``, with those parameters marked ``unaligned``, so its
compile specs give them no divisibility hint and a ``pointer_alignment`` of their element size.

A kernel parameter whose name the generated module itself uses (``tl``, ``triton``, ``float``, ``libdevice``) or
that Triton's JIT launcher takes as a launch option (``grid``, ``warmup``, ``debug``, ``num_warps``,
``num_stages``, ``num_ctas``, ``maxnreg``) is renamed with a numeric suffix (``tl_1``), in the
generated function, the extern stub and every spec alike: a spec's signature must name the
function's own parameters, so the spec cannot keep the original name. The module header says which
parameter became which; every other parameter keeps its name.

The elementwise ``n`` is recorded in the spec as ``count_of``: the tensor argument whose element
count (the product of its shape entries) the scalar carries, so a host computes it without knowing
the kernel. When the product is affine the scalar's ``dim`` is that product (the spec validator
checks it); otherwise ``dim`` is a fresh name and ``count_of`` is the only definition.

Numerics contract
-----------------
Every choice below is part of the contract; changing one later is a breaking change. Every PTX fact in
it was checked on the PTX that Triton 3.5.1 and ptxas 12.9.86 produce for sm_103, and is asserted on a GPU
by ``tests/test_gpu_numerics.py`` (the numbers in the notes are measured there).

==========================  ==========================================================  =====================================
IR                          Generated source                                            Numerics
==========================  ==========================================================  =====================================
load of a tensor            ``tl.load(p + offs, mask=mask, other=0)``                   storage dtype, never pre-cast; masked
                                                                                         lanes read 0. ``[R]``/``[R, 1]``:
                                                                                         ``tl.load(p + row)``, no mask.
``x.to(dt)``                ``v.to(tl.<dt>)``, exactly where the IR has the cast        Triton's default conversion (note 1);
                                                                                         no ``fp_downcast_rounding`` argument.
                                                                                         A float to integer cast says its
                                                                                         rounding (the next row).
                                                                                         A narrowing cast gets ``# rounding
                                                                                         point``; of a widened value, or with
                                                                                         ``fp_fusion=False``, it is inline
                                                                                         ``cvt.rn`` (note 1). Nothing else is
                                                                                         ever cast.
``x.to(i32 | i64 | u8,      a helper of inline PTX: ``cvt.rni.s32.f32`` (``rzi``,        round to nearest even, toward zero,
rounding="rn" | "rz" |      ``rmi``, ``rpi``); ``u8``: then ``max.s32``, ``min.s32``     down or up, no default; saturating
"rd" | "ru",                (to [0, 255]); ``i64`` and every wrapping cast:              (the default): clamped to the type's
saturate=True | False)``    ``cvt.rni.s64.f32``, a NaN made 0 (``testp``, ``selp``),     range, NaN 0; wrapping: the low bits
                            then the low bits (``.to()``)                                of the i64 (note 18).
``x[k]``, ``x.select(0,     row template: ``tl.load(x + k * size_R + row)`` and the      a raw read of the slice's elements
k)`` of a parameter         like (``size_R`` a runtime ``i32``); a table:                (note 19).
                            ``tl.load(x + k * N * D + i * D + cols, ...)``
``x.to(bf16,                f32 to bf16 on the bits: ``b = x.to(tl.uint32,                  the same rounding; a NaN keeps its
nan="preserve")``           bitcast=True)``, ``(b + 0x7FFF + ((b >> 16) & 1)) >> 16``,      sign and top payload bits, quiet
                            a NaN ``(b >> 16) | 0x40``                                      (note 1). ``nan="canonical"`` is the
                                                                                         default: ``0x7FFF``.
store                       ``tl.store(p + offs, v, mask=mask)``                        the IR value as is (its dtype equals
                                                                                         the target's). ``[R]``: no mask.
                                                                                         An ``Out`` or ``InOut`` parameter
                                                                                         is written once, with its final
                                                                                         version; earlier stores are elided
                                                                                         and a self-store (``x[...] = x``)
                                                                                         writes nothing.
                                                                                         A returned value is written to its
                                                                                         own output pointer (a copy).
python scalar / ``const``   a float literal written with ``repr()`` (``inf`` and           the scalar takes the dtype of the
                            ``nan`` as ``float("inf")``), an int literal, or the            tensor it meets and is rounded to it:
                            ``const`` parameter's name                                      no promotion (note 3).
``a + b``, ``a - b``,       ``a + b``, ``a - b``, ``a * b``                                 dtypes f32, f16, bf16, i32, i64.
``a * b``                                                                                    ``a * b + c`` may round once, as
                                                                                         one FMA (note 2). ``fp_fusion=False``:
                                                                                         one inline ``mul.rn`` / ``add.rn`` /
                                                                                         ``sub.rn`` each, never contracted.
``a / b``                   ``a / b``                                                       f32 only (note 4): ``div.full.f32``,
                                                                                         not IEEE division; IEEE math (note 13):
                                                                                         ``div.rn.f32``.
``i // c``, ``i % c``       ``i // c``, ``i % c`` (C's), then moved to the floor:           integers, a positive constant ``c``:
                            ``tl.where(r < 0, ...)``                                        Python's semantics, exact (note 11).
``rsqrt(x)``                ``tl.rsqrt(x)``                                                 f32 only; approximate (note 5). IEEE
                                                                                         math: ``rsqrtf`` (note 13).
``exp(x)``                  ``tl.exp(x)``                                                   f32 only; fast exp (note 5). IEEE math:
                                                                                         CUDA's ``expf`` (note 13).
``sqrt(x)``                 ``tl.sqrt_rn(x)``                                               f32 only; IEEE round to nearest, but
                                                                                         subnormals flush to zero (note 5). IEEE
                                                                                         math: ``sqrt.rn.f32`` (note 13).
``silu(x)``                 ``x * tl.sigmoid(x)``                                           f32 only; ``tl.sigmoid`` is
                                                                                         ``1 / (1 + tl.exp(-x))``. IEEE math:
                                                                                         torch's ``x / (1 + expf(-x))``.
``gelu(x)``                 ``0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))``       f32 only; exact erf form, evaluated
                                                                                         left to right (note 5).
``sin(x)``, ``cos(x)``      ``tl.sin(x)``, ``tl.cos(x)``                                    f32 only; libdevice ``__nv_sinf`` /
                                                                                         ``__nv_cosf``, built with FTZ: torch's
                                                                                         CUDA bits except ``sin`` of a
                                                                                         subnormal, which is ±0 (note 10); in
                                                                                         IEEE math that is ``x`` (note 13).
``tanh(x)``                 ``libdevice.tanh(x)`` (``from triton.language.extra import       f32 only; libdevice ``__nv_tanhf``,
                            libdevice``)                                                    built with FTZ: torch's CUDA bits
                                                                                         except on a subnormal, which is ±0;
                                                                                         in IEEE math that is ``x`` (note 14).
``erf(x)``                  ``tl.math.erf(x)``; IEEE math: ``_tf_erf_ieee(x)``              f32 only; libdevice ``__nv_erff``
                                                                                         (gelu's; FTZ, not CUDA's erff). IEEE
                                                                                         math: CUDA's ``erff`` without FTZ
                                                                                         (CUDA 12.8 to 13.1; note 15).
``sin_approx(x)``,          a module-level helper of one PTX instruction (inline asm):      CUDA's ``__sinf``, ``__frcp_rn``,
``rcp(x)``, ``fma(a, b,     ``sin.approx.f32``, ``rcp.rn.f32``, ``fma.rn.f32``,         ``fmaf`` (one rounding, whatever
c)``, ``neg(x)`` / ``-x``   ``neg.f32`` (``.bf16``, ``.f16``) with ``testp`` / ``selp``  ``fp_fusion``) and ``-x`` (a NaN gives
                            for a NaN; integers ``-v``
                                                                                         the canonical NaN), in either math;
                                                                                         f32 (``neg``: f16, bf16, i32, i64
                                                                                         too). Note 16.
``sum(x, axis=-1)``         ``tl.sum(tl.where(mask, x, 0.0), axis=0)``                      f32 only. The reduction is over all
                                                                                         ``BLOCK_D`` lanes; masked lanes are
                                                                                         set to 0 explicitly (intermediates
                                                                                         such as ``exp(x)`` or ``x - mean``
                                                                                         are not 0 there). The order is
                                                                                         Triton's, and depends on
                                                                                         ``num_warps`` (note 6).
``sum(x, axis=-1,           a module-level helper per order: ``lanes`` chains of            f32 only; the order of every add is
order=...)``                ``add.rn`` (``fma.rn``: ``fma=True``) over the columns,      the call's (``"sequential"`` or
                            then ``tl.split`` steps of the tree over the lanes          ``tilefish.tree(...)``), the same at
                                                                                         every ``num_warps`` (note 17).
``mean(x, axis=-1)``        ``tl.sum(tl.where(mask, x, 0.0), axis=0) / D``                  f32 only; ``D`` is the constexpr:
                                                                                         ``sum * f32(1 / D)`` (note 4).
``max(x, axis=-1)``         ``tl.max(tl.where(mask, x, float("-inf")), axis=0)``            f32 only; masked lanes are ``-inf``
                                                                                         (``float("-inf")`` compiles: JIT and
                                                                                         AOT, ``0fFF800000`` in the PTX).
``a == b``, ``a != b``,     ``a == b`` ... ``a >= b``; a scalar of a bf16, f16, u8 or       a mask (``bool_``, Triton's ``int1``).
``a < b``, ``a <= b``,      i64 operand is a typed constant, ``tl.cast(c, tl.bfloat16)``,   IEEE: ``-0.0 == +0.0``, a NaN is
``a > b``, ``a >= b``       and a literal is written already rounded to that dtype         unordered (every comparison with one is
                                                                                         false except ``!=``, which is true).
                                                                                         Exact; the scalar is rounded to the
                                                                                         operand's dtype first (note 7).
``m & n``, ``m | n``,       ``m & n``, ``m | n``, ``m ^ n``, ``~m``                         masks only (no bitwise ops on numbers)
``m ^ n``, ``~m``
``where(m, a, b)``          ``tl.where(m, a, b)`` (scalars typed as for a comparison)       a select: the bits of the chosen
                                                                                         operand, unchanged (NaN payloads,
                                                                                         ``-0.0``; note 7).
``iota(x, axis)``           row template: ``row`` (``[R, 1]``, ``[R]``) or ``cols``         ``i32``, exact; ``dtype=i64`` adds
                            (``[D]``); elementwise: ``offs``                                ``.to(tl.int64)``.
``mask.to(dt)``             ``m.to(tl.<dt>)``                                               exactly 0 or 1 (zero extension,
                                                                                         ``uitofp``); a cast *to* a mask is
                                                                                         refused (write ``x != 0``).
``argmax(x, -1, dtype=)``   row template only: ``tl.reduce`` of (value, column) pairs       exact and order-free: the first NaN,
``argmin(x, -1, dtype=)``   with a combine that implements the order; a row of 1025 to      else the first extreme value
                            16384 columns is walked in chunks of 1024 (note 8)              (``-0.0 == +0.0``); torch's rule.
                                                                                         ``dtype`` is ``i32`` or ``i64``.
``gather_rows(x, idx,       ``ok = (i >= 0) & (i < x_rows)``, then                          a raw copy (the loaded bits are the
oob="zero" | "clamp")``     ``tl.load(x + i * D + cols, mask=mask & ok, other=0)``;         stored bits). Out of range: a row of
                            ``clamp``: ``i = tl.minimum(tl.maximum(i, 0), x_rows - 1)``     ``+0.0`` (``zero``) or row 0 / N - 1
                            (an ``[N]`` table: ``x + i``)                                   (``clamp``); note 9.
``scatter_rows(dst, idx,    ``tl.store(dst + i * D + cols, src, mask=mask & ok)``           a raw copy; a row whose index is out
src, oob="skip")``                                                                          of range is not written; indices in
                                                                                         range must be distinct (note 9).
==========================  ==========================================================  =====================================

Reductions are f32 only because Triton accumulates a narrower dtype in the narrow dtype, which
the IR does not say; cast to f32 first (the README kernel does). Every SSA value is one local
(``v0``, ``v1``, ... in IR order: the loaded parameters first, then one per node), and a comment
marks each rounding point.

Notes (the facts behind the table):

1. **Casts.** Float to ``f16``/``bf16`` is ``cvt.rn.f16x2.f32`` / ``cvt.rn.bf16x2.f32``: round to nearest even,
   overflow to infinity (65520 is the first f32 that becomes ``inf`` in f16), NaN stays NaN; the same bytes as
   torch's cast. Float to ``e4m3`` / ``e5m2`` is ``cvt.rn.satfinite.e4m3x2.f32`` / ``cvt.rn.satfinite.e5m2x2.f32``:
   round to nearest even and **satfinite**, saturating at the largest finite value (448, 57344) with
   infinities included, where torch overflows to NaN (1000 to ``e4m3``) or infinity (1e5 to ``e5m2``); NaN stays
   NaN. The ``reference`` mode models ``satfinite``, not torch's cast. Integer to float and integer to integer
   casts are the plain conversions. A **float to integer cast says its rounding and saturation** (note 18) and is written out as
   inline PTX, not Triton's ``.to()``, because of what Triton does, measured for a hand-written kernel on sm_103: ``i32`` is
   ``cvt.rzi.s32.f32`` (truncation toward zero, saturating, NaN is 0), ``i64`` is ``cvt.rzi.s64.f32`` (the same, but NaN is
   ``-2**63``), and ``u8`` is
   ``cvt.rzi.u16.f32`` followed by keeping the low byte: truncated and clamped to [0, 65535] (negative and NaN
   give 0), then reduced modulo 256, so ``-1.5`` is 0, ``255.5`` is 255, ``256`` is 0, ``300`` is 44, ``1000`` is
   232 and anything from 65535 up is 255; it neither saturates at 255 nor wraps like C. torch differs again
   (``-1.5`` to ``u8`` is 255).
   **A NaN in the f32 to bf16 cast** has two declared modes. ``x.to(bf16)`` is ``x.to(bf16, nan="canonical")``:
   ``cvt.rn.bf16.f32``, which gives the canonical NaN ``0x7FFF`` for every NaN. ``x.to(bf16, nan="preserve")`` is the integer
   round to nearest even on the bit pattern (``r = (b + 0x7FFF + ((b >> 16) & 1)) >> 16`` on the u32 bits), and a NaN keeps its
   sign and its top payload bits with the quiet bit set (``(b >> 16) | 0x40``: ``0xFFC00000`` becomes ``0xFFC0``, ``0x7F800001``
   becomes ``0x7FC0``). The two modes give the same bits on every other f32 (subnormals and infinities included); over all 2**32
   inputs they differ on 16 646 142, every one a NaN whose upper half is not ``0x7FFF`` (``tests/test_gpu_casts.py``). An f32 that
   comes out of GPU arithmetic is never such a NaN (``add`` and ``mul`` produce only ``0x7FFFFFFF``), so the mode matters for a cast
   of an input. torch's own cast of a NaN is a third answer that depends on where it runs (``0x7FFF`` on CUDA, ``0x7FC0`` or ``0xFFFF``
   on the CPU); the ``reference`` mode computes the declared mode on the bits, so it is byte-identical to the binary in both.
   Only the f32 to bf16 cast has ``nan=``.
   **A narrowing cast of a widened value** (``x.to(f32).to(bf16)``, and one of a ``where`` that selects a widened value, through any number of
   them) is written out as inline PTX in every kernel, ``cvt.rn.bf16.f32`` / ``cvt.rn.f16.f32`` (a 16-bit source widened to f32 first, exactly):
   the instruction ``.to()`` compiles to, out of the reach of the folds of MLIR and LLVM, which take ``fptrunc(fpext(x))`` for ``x`` and narrow a
   select of widened values into a select of the values, keeping an input NaN's payload where the cast gives ``0x7FFF``. Measured on sm_103 (RFC
   0001, experiment E5): with ``.to()``, ``x.to(f32).to(bf16)`` in one kernel kept the payload of every input NaN (254 of 254 bf16 NaNs, 2046 of
   2046 f16 NaNs). Every other cast of a kernel with ``fp_fusion=True`` is ``.to()``; with ``fp_fusion=False`` every narrowing float cast is
   inline PTX (note 2). The inline cast is ``.to()``'s bits on every f32 (``tests/test_gpu_fusion.py``), and gives the canonical NaN for every
   NaN of a widened value (``tests/test_gpu_roundings.py``).
2. **FMA contraction.** A generated kernel asks Triton for ``enable_fp_fusion=True`` (Triton's default, written
   into the spec), so ``a * b + c`` **may round once**: it is one FMA, ``fma.rn.f32`` for f32 and ``fma.rn.bf16x2``,
   a native packed bf16 FMA, for bf16. With ``@tilefish.kernel(fp_fusion=False)`` every float ``+``, ``-`` and ``*`` (f32, bf16, f16)
   **rounds on its own** ("strict"), and that is written out, not left to the compile option: each is one scalar instruction of inline PTX with
   an explicit rounding (``mul.rn.f32``, ``add.rn.bf16``, ...), and so are the last multiply of ``silu`` and the multiplies and add of ``gelu``;
   every narrowing float cast is inline ``cvt.rn`` (note 1). That is the code of a fused region (note 20): a kernel with ``fp_fusion=False`` is
   lowered as one, but for its header. The compile option alone (Triton's ``--fmad=false``) does not keep it: on sm_100 and later LLVM packs two
   lanes' f32 operations into ``mul.rn.f32x2`` / ``add.rn.f32x2``, and ptxas 12.9 contracts such a pair whatever ``--fmad`` says, into ``FFMA2``
   on sm_100 and sm_103 and into a scalar ``FFMA`` on sm_120 (which splits the packed PTX op). What LLVM packs depends on the kernel (measured
   with Triton 3.5.1: ``x * w + b`` of a row of widened bf16 values with ``[D]`` operands is packed and contracted on all three, ``a * b + c``
   of three f32 loads of one length is not, and on sm_120 a row sum of squares is); a scalar ``.rn`` instruction is never contracted (the random
   graphs of ``tests/test_gpu_fusion.py`` found it first: 33 of 1536 elements of one graph one ULP off). It is also **checked on the binary**:
   every kernel built with ``enable_fp_fusion=False`` is read back (:mod:`tilefish._fpcontract`, at build), and one whose SASS has more fused
   multiply-adds than the SASS of its PTX with the multiplies taken out is refused (``tests/test_gpu_fpcontract.py`` builds every case of the
   lowering with ``fp_fusion=False`` for sm_90, sm_100, sm_103 and sm_120). The two results differ in 23% of random elements (22.9% in bf16,
   23.4% in f32), and the contracted one is exactly the correctly rounded value of the exact product and sum. For the tail of layernorm,
   ``(x - mean) * rstd * w + b`` in f32, contraction changes 25.7% of the elements and the strict build matches a
   reference that rounds every operation exactly. On the goldens contraction changes the bytes of the ones that
   have a contractable ``a * b + c`` (layernorm: 3 to 8 of 197376 elements, one bf16 ULP; the README rmsnorm: 1 to 2
   of 657920); softmax, gelu and swiglu come out byte-identical either way. The explicit FMAs inside libdevice's
   ``erf`` stay with ``fp_fusion=False``; Triton's own reductions (``sum`` without ``order``, ``mean``) add as Triton compiles them, the
   check included.
3. **Scalars.** Triton 3.5.1 keeps the dtype of the tensor: a python float or a constexpr float combined with
   a bf16 tensor is bf16 (``tl.static_assert`` in a compiled kernel says so), and the scalar is rounded to
   bf16 first (0.1 becomes 0.10009765625). Nothing is promoted to f32: ``x * 0.1 + eps`` on bf16 is a single
   ``fma.rn.bf16x2`` (or ``mul.rn.bf16x2`` and ``add.rn.bf16x2`` without contraction), with no ``cvt`` to f32.
   The one exception is ``/``, which Triton does in f32 for bf16 and f16 and returns as f32; that is why a
   division of those dtypes is rejected.
4. **Division.** ``a / b`` is ``div.full.f32``, never ``div.rn.f32``: the numerator times an approximate
   reciprocal, at most 2 ULP from the correctly rounded quotient (measured: 2 ULP, 29% of random elements
   differ, relative error below 1.8e-7). With a constant divisor (``x / D`` with ``D`` a constexpr or a
   literal) the PTX is the same ``div.full.f32`` (Triton does not strength-reduce it), but the machine code
   behind it is a multiply by the f32 reciprocal: the result is bit-identical to ``x * f32(1 / c)`` for
   every constant tried (4M inputs each), and differs from correctly rounded division in 15% to 58%
   of random inputs (33% for 768, 58% for 1000); only a power of two is exact. ``mean`` is ``sum * f32(1 / D)``
   for the same reason.
5. **Math functions.** ``tl.sqrt_rn`` is ``sqrt.rn.ftz.f32``: the correctly rounded square root for every normal
   input (the same bytes as torch's), but a subnormal input is treated as zero, so ``sqrt(1e-40)`` is 0, not
   1e-20; it is not ``sqrt.rn.f32``. ``tl.rsqrt`` is ``rsqrt.approx.ftz.f32`` (up to 2 ULP, 22% of inputs off the
   correctly rounded value; a subnormal input is zero). ``tl.exp`` is ``ex2.approx.f32`` of
   ``x * 1.4426950216293335`` (log2 e as an f32): the product is rounded before the exponentiation, so the
   error grows with ``|x|`` (3 ULP for ``|x| < 2``, 27 ULP up to ``|x| = 40``). ``tl.math.erf`` (also
   ``tl.erf``) exists and links: libdevice's ``__nv_erff`` is inlined (no ``call`` is left in the PTX) as a
   polynomial of ``fma.rn.ftz.f32`` and ``ex2.approx.ftz.f32``, which flush subnormals; its FMAs are explicit, so
   they do not depend on ``enable_fp_fusion``. The exact gelu cancels for large negative inputs (``1 + erf`` is
   a difference of two numbers near 1), so its relative error there is large (128 bf16 ULP on values near
   zero), though its absolute error stays below 1.6e-5.
   **libdevice and flush to zero.** Triton always links libdevice built with ``__CUDA_FTZ = 1``; there is no option to
   change it, and tilefish adds none. The libdevice functions a lowered kernel calls are ``__nv_erff`` (``gelu``),
   ``__nv_sinf`` and ``__nv_cosf`` (note 10). Against the same functions built without FTZ (what a CUDA C++ or Rust
   kernel and torch's CUDA kernels get), over all 2**32 f32 inputs: ``erff`` and ``sinf`` differ only on subnormal
   inputs, ``cosf`` on none, and ``expf`` on none (``tilefish.exp`` is not ``expf``: it is ``ex2.approx``, above).
6. **Reduction order and ``num_warps``.** The tree a reduction adds in depends on how many warps the block has,
   so the same kernel on the same data is not byte-identical across ``num_warps`` (129 rows with D = 96: 17 to 59
   rows differ from the 1-warp result in f32, depending on the kernel; with D = 1000: 28 to 125), though
   always within a few ULP of the others and of torch's sum. That is why each bucket of a bucketed ``D`` has the
   warps of its own ``BLOCK_D``: the ``D = 96`` binary has 1 warp (128 lanes) whichever other buckets are declared
   and has the same bits as a kernel whose ``D`` is 96 alone, instead of the warps of the largest bucket (which
   made a ``D = 96`` bucket next to a ``D = 2048`` one run with 8 warps and differ from the 1-warp result in 25
   of 129 rows). A spec fixes ``num_warps``, so ``jit``, ``replay`` and a generated host all agree, and
   ``num_warps=N`` pins one value for every bucket.

7. **Comparisons, masks and ``where``.** A comparison gives a mask (``tilefish.bool_``), which is a value inside the body only: it
   has no arithmetic and no storage type (a parameter or a result is never a mask; store one with ``.to(u8)``, read it back with
   ``m != 0``). Masks combine with ``& | ^ ~``. Float comparisons are IEEE 754 ones, the same as torch's: ``==`` is ``oeq``, ``!=`` is
   ``une`` (so ``x != x`` is the NaN test), ``<`` ``<=`` ``>`` ``>=`` are ordered, and ``-0.0 == +0.0``. The operands have one dtype;
   a python number or a ``const`` takes it and is **rounded to it first**, as in arithmetic (note 3). Triton itself does not do that
   for a comparison: it widens a bf16 or f16 tensor to f32 and compares it with the f32 literal unrounded, so ``x > 0.1`` with
   ``x = bf16(0.1) = 0.10009765625`` is true in a hand-written kernel and false in torch. The lowering therefore writes such a scalar
   as a typed constant, ``tl.cast(0.10009765625, tl.bfloat16)``, with a literal already rounded to the dtype (through f32, as torch
   rounds a python float), so the comparison is exact whatever Triton widens to; a ``const`` is ``tl.cast(name, tl.bfloat16)``
   (rounded by Triton from the f32 constant, also through f32, except for a value outside the normal f32 range, which Triton keeps
   in f64 and rounds once). A ``u8`` or ``i64`` scalar is typed the same way, so nothing is promoted. ``where`` is a select
   (``selp``): the result has the bits of the operand it picks, a NaN's payload and the sign of a zero included. ``iota`` and the
   casts of a mask (0 or 1) are exact. The ``reference`` mode does all of these in torch with the scalar rounded the same way, so it
   is byte-identical to the binary.
8. **``argmax`` and ``argmin``.** The result is the index (``dtype=i32`` or ``i64``, which the call must say) of the winner of a strict
   total order on (value, index): a NaN beats every number (``+inf`` included, for ``argmax`` and ``argmin`` alike), two NaNs or two
   equal values (``-0.0`` and ``+0.0`` are equal) go to the lower index, otherwise the larger (``argmax``) or smaller (``argmin``) value
   wins. So the first NaN wins, then the first extreme value; a row of ``-inf`` gives 0 for ``argmax``. This is torch's documented rule
   (the index of the first maximal value, a NaN propagating), and what ``torch.argmax`` / ``torch.argmin`` return on CUDA, checked on
   adversarial rows by ``tests/test_gpu_argmax.py``. It is not ``tl.argmax``'s: its combine (``v1 > v2 or (v1 == v2 and i1 < i2)``) lets
   a NaN win only from one side, so its answer depends on the tree. A bf16 or f16 value is widened to f32 (exact), an integer is
   compared as it is (``u8`` unsigned). The lowering is a module-level combine of (value, index) pairs that implements the order
   (``_tf_argmax_float`` and the like), applied by ``tl.reduce``; a masked lane offers (``-inf`` / ``+inf`` or the integer type's bound,
   ``2**31 - 1``) and loses to every lane. Because the order is total, the combine is associative and commutative and the result does
   not depend on the reduction tree, on ``num_warps`` or on how the row is cut. A row of more than ``1024`` columns (and at most
   ``16 * 1024``) whose kernel has no ``sum``, ``mean`` or ``max`` and nothing per element that depends on the result is walked in
   chunks of ``BLOCK_D = 1024`` columns, unrolled, in ascending order: each lane folds its column of every chunk into a carried pair,
   then one ``tl.reduce`` of the carries, with 2 warps; everything per element in the kernel is computed chunk by chunk in the same
   loop. Measured on a B300 MIG slice, GPU time of a captured plan, bf16 rows of a length between 4096 and 8192: 30.8 microseconds
   for 2048 rows chunked, 70.4 as one 8192-wide block (the same combine), 55.9 for ``torch.argmax``; 6.2 / 12.3 / 18.5 for 256 rows;
   6.0 / 5.9 / 18.5 for 64 rows.
   A longer row is one block (faster there: 32768 columns). Only the row template has them, along the last axis.
9. **Gathers and scatters of rows.** ``gather_rows(x, idx, oob=...)`` is ``x[idx]`` along the first axis and ``scatter_rows(dst, idx, src,
   oob="skip")`` is ``dst[idx] = src`` in place (an index copy). Both are raw copies: a value is loaded and stored with no arithmetic, so a
   NaN keeps its payload and a zero its sign. The table ``x`` is a read-only ``Tensor[...]`` parameter and ``dst`` an ``InOut`` parameter
   that nothing else in the kernel reads or writes (rules ``gather.table``, ``scatter.target``): their rows are addressed by the index,
   not by the program, so the template does not fit them to its shapes. A table is ``[N, D]`` with the row template's ``D`` (one program
   per index, which reads or writes one row; ``idx`` is ``[R]``) or ``[N]`` (any template; element by element). An index is an ``i32``,
   ``i64`` or ``u8`` tensor of device data, which nothing checks before the kernel reads it, so what an index outside ``[0, N)`` does is
   part of the call and has no default: ``"zero"`` loads a row of ``+0.0`` (all bits clear; the load is masked, nothing out of range is
   read), ``"clamp"`` loads row 0 or ``N - 1``, and a scatter's ``"skip"`` writes nothing for that row. A negative index is out of range
   (no wrap-around). There is no unchecked mode: the check is one compare per row. ``N`` is a parameter after the template's own
   (``<x>_rows``: a ``tl.constexpr`` when ``N`` is static or bucketed, a runtime ``i32`` when it is dynamic). The offset is ``i32``: the
   index is checked (or clamped) first, and ``N * D`` is held to ``2**31 - 1`` elements like every tensor's size (rule ``index-range``, a
   ``max_elements`` precondition when it is dynamic), so ``i * D + cols`` fits. The indices in range of one scatter must be distinct:
   two programs that write one row race, and which bytes land is not defined (torch's ``index_copy_`` does not define it either).
   Rows of ``dst`` that no index names keep their bytes.
10. **``sin`` and ``cos``.** ``tl.sin`` and ``tl.cos`` are libdevice's ``__nv_sinf`` and ``__nv_cosf``, inlined (no ``call`` is left: a
    polynomial of ``fma.rn.ftz.f32`` after a range reduction), the same PTX as ``libdevice.sin`` / ``libdevice.cos``. Triton links
    libdevice with ``__CUDA_FTZ = 1``, so the only inputs on which they differ from the non-FTZ library (torch's CUDA ``sinf`` /
    ``cosf``) are the subnormals, where ``sin(x)`` is ``±0`` (the sign of ``x``) instead of ``x``; ``cos`` is the same on every input. The
    reference is torch's with that flush, so it is byte-identical to the binary (``tests/test_gpu_trig.py``). f32 only.
11. **``//`` and ``%`` of integers.** Index arithmetic: an ``i32``, ``i64`` or ``u8`` tensor divided by a positive constant (a python int,
    an integer ``const``, a static dimension; a tensor divisor could be zero and is refused). The semantics are Python's: the quotient
    rounds toward minus infinity and the remainder is in ``[0, c)`` (``-7 // 2 == -4``, ``-7 % 2 == 1``). Triton's ``//`` and ``%`` are C's
    (``sdiv`` truncates, ``srem`` takes the dividend's sign), so for a signed type the lowering computes C's remainder and moves a
    negative one: ``tl.where(r < 0, r + c, r)`` and ``tl.where(r < 0, i // c - 1, i // c)``; an unsigned type needs nothing. Exact, with no
    overflow (``r + c`` and ``i // c - 1`` are taken only where ``r < 0``). The reference is ``torch.div(..., rounding_mode="floor")`` and
    ``torch.remainder``.
12. **Broadcasts along any axis** (the elementwise template). The kernel's shape ``S`` (rank ``r``) is the one every tensor broadcasts to; an
    operand of shape ``T`` is aligned on the last axes, and along each axis it either *has* the axis (its size is ``S``'s) or repeats (size 1, or
    no such axis). An operand whose axes are ``S``'s last ones is read at ``o % count`` (above). Any other is read at the offset of the lane's
    coordinates: for each maximal run ``[a, b]`` of the axes it has, the lane's index along the run taken as one axis,
    ``(o // prod(S[b+1:])) % prod(S[a:b+1])`` (no division for the last axes, no remainder for the first), times the operand's own stride after the
    run (the product of the sizes it has after ``b``), summed over the runs: ``[H, 1, L]`` of ``[H, L, L]`` is ``(o // (L * L)) * L + o % L``, and
    ``[K, T, 1]`` of ``[K, T, D]`` is ``o // D``. ``iota(x, axis)`` is the lane's coordinate along that kernel axis, ``(o // prod(S[k+1:])) % S[k]``. A
    static size is written as a number; a symbolic one is a runtime ``i32`` parameter of the dimension it is (``size_L``, after the template's own and
    the broadcast counts), which a host derives from the shapes like any dimension scalar, so one binary serves every size. Exact: integer
    arithmetic on ``i32``, every intermediate below ``n``, which is below ``2**31``. A broadcast operand is never written; a gather or a scatter of a
    broadcast shape is not lowered (rule ``elementwise.broadcast-index``).
13. **IEEE math** (``@tilefish.kernel(math="ieee")``, or ``math="ieee"`` at one op: ``tilefish.div(a, b, math="ieee")``, ``tilefish.exp(x,
    math="ieee")``, ...; an op's own ``math=`` wins over the kernel's, ``"fast"`` included). The f32 math of CUDA C++ compiled without fast math and
    without flush to zero (nvcc's defaults ``-ftz=false -prec-div=true -prec-sqrt=true``, what torch's CUDA kernels are built with), written out
    as the inline PTX nvcc emits (NVRTC 12.8, sm_100), because Triton links libdevice with ``__CUDA_FTZ = 1`` and emits ``.ftz`` for its own
    ``div_rn`` and ``sqrt_rn``. Each op is a module-level helper (``_tf_exp_ieee``) whose ``add`` and ``mul`` carry ``.rn``, which rounds as the
    plain form does and keeps ``ptxas`` from contracting them with an instruction outside the block:

    * ``a / b`` is ``div.rn.f32``: IEEE 754 division, correctly rounded, subnormals kept; a constant divisor too (it is divided by, not
      multiplied by its reciprocal; torch's ``x / 3.0`` of a python scalar multiplies by ``f32(1 / 3)``, its ``x / tensor`` divides);
    * ``exp(x)`` is CUDA's ``expf``: ``j`` = ``x * log2(e)`` rounded down to an integer through ``fma.rm`` (with the bias ``12582913``), ``r = x *
      log2(e) - j`` in two ``fma.rn`` (``log2(e)`` as ``0x3FB8AA3B`` plus ``0x32A57060``), ``ex2.approx.ftz(r)`` (``r`` is in ``[-1, 1]``: nothing to
      flush) times ``2**j`` built in the exponent bits, one ``mul.rn``: at most 2 ULP, subnormal results kept (``x`` from about ``-103.97`` to
      ``-87.34``, which the default math's ``ex2.approx`` of ``x * log2(e)`` and libdevice's FTZ ``expf`` give as 0);
    * ``silu(x)`` is torch's ``x / (1 + expf(-x))`` as nvcc compiles it: the negation folded into the constants of ``expf`` (exact), ``2**r * 2**j +
      1`` one ``fma.rn`` (nvcc contracts the multiply that ends ``expf`` with the add, ``-fmad=true``), then ``div.rn.f32``. A body that writes
      ``x / (1.0 + tilefish.exp(-x))`` instead rounds the multiply first: tilefish never contracts across an op of IEEE math;
    * ``sqrt(x)`` is ``sqrt.rn.f32`` (correctly rounded, a subnormal argument kept), ``rsqrt(x)`` is CUDA's ``rsqrtf``, ``rsqrt.approx.f32``
      without flush to zero;
    * ``sin(x)`` is libdevice's ``sinf`` (built with FTZ, which differs from the library without it only on a subnormal argument, note 10) with
      that argument passed through as its own sine: ``tl.where(tl.abs(x) < 2**-126, x, tl.sin(x))``; ``cos`` is the same in both maths;
    * ``tanh(x)`` is libdevice's ``tanhf`` with a subnormal argument passed through likewise (note 14).

    Measured on every one of the ``2**32`` f32 bit patterns (``tests/test_gpu_ieee.py``; torch 2.9.1 on a B300 MIG slice, sm_103, built with CUDA 12.8
    and with 13.0, and on an RTX 5090, sm_120, built with CUDA 13.0): ``exp``,
    ``sqrt``, ``rsqrt``, ``sin`` and ``silu`` are torch's CUDA bits on every input, NaNs, infinities, zeros and subnormals included (the default math
    differs on 129 346 491 inputs for ``exp``, 146 046 470 for ``silu``, and on the 16 777 214 subnormals and nothing else for ``sqrt``, ``rsqrt`` and
    ``sin``); ``/`` is torch's division of tensors on three sets of 2**26 random bit-pattern pairs, every subnormal as the numerator and as the divisor and the special values pairwise (the default math's
    differs in 49 229 767 of those 234 881 216 pairs on sm_103, 49 247 968 on sm_120). Other ops (``gelu``'s ``erf``, the reductions, the casts) are the same in both maths, and so is ``mean``
    (``sum * f32(1 / D)``, as torch's ``mean`` multiplies by ``1 / N``). With ``fp_fusion=False`` too, a body of these ops rounds where torch's
    eager ops round, so it is torch's eager composition byte for byte. The ``reference`` mode in IEEE math is torch's own op (``torch.div`` of
    tensors, ``torch.exp``, ``torch.nn.functional.silu``, ...).
14. **``tanh``.** ``tilefish.tanh(x)`` (f32) is libdevice's ``__nv_tanhf``, called by name (``libdevice.tanh`` of ``triton.language.extra``, which
    the generated module imports when the body has a ``tanh``) and inlined; Triton links libdevice with ``__CUDA_FTZ = 1``, so its ``fma``,
    ``mul`` and ``add`` are ``.ftz``. The function is CUDA's ``tanhf``: for ``|x| >= 0.6``, ``1 - 2 / (2**(2 |x| log2 e) + 1)`` from
    ``ex2.approx.ftz`` and ``rcp.approx.ftz`` (1 from ``|x| >= 9.0109``), the sign of ``x`` put back on the bits; below, the odd polynomial
    ``x + x * x**2 * p(x**2)`` of four ``fma``. Measured over every one of the ``2**32`` f32 bit patterns against torch's CUDA ``tanh`` (torch
    2.9.1 on a B300 MIG slice built with CUDA 12.8 and with 13.0, and on an RTX 5090 with 13.0; ``tests/test_gpu_tanh.py``): the bits are torch's on every input but the 16 777 214 subnormals, where
    the binary gives a zero of the argument's sign (torch: the argument itself), as ``sin`` does (note 10). In IEEE math (note 13) a subnormal
    argument is passed through, ``tl.where(tl.abs(x) < 2**-126, x, libdevice.tanh(x))``, and the bits are torch's on every input. The
    ``reference`` mode models the flush in the default math (``torch.where(|x| < 2**-126, x * 0.0, torch.tanh(x))``) and is torch's ``tanh`` in
    IEEE math: byte-identical to the binary in both. ``tanh.approx.f32`` (one instruction, about ``2**-11`` relative error) is not used.
15. **``erf``.** ``tilefish.erf(x)`` (f32) is ``tl.math.erf``, libdevice's ``__nv_erff`` as Triton links it (with FTZ): the ``erf`` of ``gelu``
    (note 5). That library is not CUDA's: over every f32 it differs from CUDA's ``erff`` on 24 071 628 inputs (by 6e-8 at most from torch's ``erf``
    on N(0, 3) data). In IEEE math (note 13) it is CUDA's ``erff`` without flush to zero, a module-level helper of the inline PTX NVRTC emits for it
    (``_tf_erf_ieee``: a polynomial in ``x * x`` below ``|x| = 1.0029547``, one in ``|x|`` above that ends in ``1 - ex2.approx.ftz``, the sign of
    ``x`` put back with ``copysign``; nvcc's one branch is a select), every ``add`` and ``mul`` ``.rn``. Measured over every one of the ``2**32`` f32
    bit patterns (``tests/test_gpu_primitives.py``, sm_103): the bits of CUDA 12.8's ``erff`` (NVRTC, and torch's ``erf`` built with CUDA 12.8),
    of 12.9's, 13.0's and 13.1's, which are the same function; CUDA 13.2 changed ``erff`` (other coefficients and threshold), which differs on
    353 987 744 inputs by one ULP. The CUDA math guard records this (``docs/numerics.md``). ``gelu`` keeps libdevice's ``erf`` in both maths (its
    bits are part of its contract); CUDA's exact gelu in IEEE math is written with ``tilefish.erf``. The ``reference`` mode is ``torch.erf``:
    byte-identical in IEEE math where torch is built with CUDA 12.8 to 13.1, within ``2**-22`` in the default math (asserted on normal data).
16. **One-instruction primitives**, each a module-level helper of one inline PTX instruction, in either math and whatever ``fp_fusion`` says
    (an inline-asm result is never contracted with anything): ``tilefish.sin_approx(x)`` is ``sin.approx.f32``, CUDA's ``__sinf`` (the hardware's
    sine: absolute error below ``2**-20.5`` on ``[-pi, pi]``, measured ``3.6e-7``, growing outside; a subnormal argument gives a zero of its sign,
    with or without ``.ftz``, which give the same bits on every f32); ``tilefish.rcp(x)`` is ``rcp.rn.f32``, the correctly rounded ``1 / x`` with
    subnormals kept, CUDA's ``__frcp_rn`` and its ``1.0f / x`` without fast math, and ``torch.reciprocal``; ``tilefish.fma(a, b, c)`` is
    ``fma.rn.f32``, ``a * b + c`` rounded once with subnormals kept, CUDA's ``fmaf`` (operands broadcast like ``+``'s; a scalar is rounded to f32);
    ``tilefish.neg(x)``, which ``-x`` is, is CUDA's negation: the sign of every number flipped, zeros and infinities included, where Triton's
    ``-x`` is ``0 - x`` (``+0.0`` for ``+0.0``), and the canonical NaN (``0x7FFFFFFF``, ``0x7FFF``) for every NaN, which a flip of the sign bit does
    not give. The NaN is written out, in every kernel: ``testp.notanumber.f32`` (``setp.nan.bf16`` / ``setp.nan.f16``) and ``selp`` of the canonical
    NaN and ``neg.f32`` (``neg.bf16``, ``neg.f16``). ``neg.f32`` alone gives the canonical NaN only while ptxas keeps it an instruction: folded into
    its consumer as a negated operand (a ``where`` of ``-x``: ``FSEL`` of ``-R``) it flips the sign of the NaN instead (``0xFFFFFFFF``; found by
    the random graphs of ``tests/test_gpu_fusion.py``, 61 of 1536 elements of one graph), and the select's other operand is unchanged either
    way; ``tests/test_gpu_roundings.py`` checks ``-x`` alone, in a ``where`` on either side and twice, on every f32 NaN and every bf16 and f16
    pattern, in a kernel, with ``fp_fusion=False`` and in a fused region. ``-x`` of an ``i32`` or ``i64`` is ``0 - x`` (two's complement). Measured
    against CUDA C++ compiled by NVRTC and against PTX on every f32 (and every bf16 and f16 for ``neg``), and ``fma`` on 3 sets of ``2**24``
    operand triples (bit patterns, cancellations ``c ~ -a * b``, the subnormal range, every combination of the special values):
    ``tests/test_gpu_primitives.py``. The ``reference`` mode is exact for ``rcp`` (``torch.reciprocal``), ``neg`` and ``fma`` (an f64 product and
    sum, rounded to odd, then to f32; NaNs canonical), and ``torch.sin`` within the bound for ``sin_approx``.
17. **Ordered sums.** ``tilefish.sum(x, axis=-1, order=...)`` (f32, the last axis of ``[R, D]``, the row template) adds in the order the call
    says, which is what a CUDA kernel's bytes depend on; without ``order`` it is Triton's tree (note 6). ``order=tilefish.tree(lanes, offsets,
    vector)``: ``lanes`` partial sums, each a chain from ``+0.0``: lane ``j`` takes the columns ``c * lanes * vector + j * vector + v`` that are
    below ``D``, in the order ``(c, v)`` (``c = 0, 1, ...``, ``v`` from 0 to ``vector - 1``), each step ``acc + x`` rounded once (``add.rn.f32``);
    a lane with no column keeps ``+0.0``. Then, for each ``o`` of ``offsets`` in turn, every lane ``j`` adds lane ``j + o`` (``p[j] = p[j] + p[j +
    o]``, ``add.rn``), and the sum is lane 0's. ``offsets`` is a permutation of the powers of two below ``lanes`` (default: ``lanes / 2``, ...,
    1, the halving tree of a shared-memory reduction ``s[t] += s[t + o]``); a warp-shuffle tree (``__shfl_down_sync`` or ``__shfl_xor_sync`` with
    16, ..., 1: IEEE addition commutes, so lane 0 is the same) followed by one over the 4 warps of a 128-thread block,
    ``(w0 + w2) + (w1 + w3)``, is ``offsets=(16, 8, 4, 2, 1, 64, 32)``. ``order="sequential"`` is ``tree(lanes=1)``: one chain over the row, left
    to right from ``+0.0``. ``fma=True`` (with an order, of a product ``a * b`` only) makes each step of a chain ``acc = fma(a, b, acc)``, one
    rounding: what nvcc compiles ``acc += a * b`` to. Lowered as a module-level helper per order: a loop over a lane's steps that takes each
    step's terms with ``tl.gather`` (``tl.where`` keeps a lane's value where the column is at or past ``D``: skipped, not added as ``+0.0``, so a
    chain that holds ``-0.0``, which an fma of tiny products of opposite signs gives, keeps it as a CUDA loop that stops at ``D`` does), then one
    ``tl.reshape`` / ``tl.permute`` / ``tl.split`` per offset (data movement: exact) and one ``add.rn``; ``tl.reduce`` with a combine that keeps its
    first operand takes lane 0. Nothing in it depends on ``num_warps`` or on Triton's layouts: the same bytes at 1, 2, 4, 8 and 16 warps.
    Measured byte for byte against CUDA C++ kernels of each order (one thread per row; 256 threads and a halving tree; 128 threads of 8 columns
    with ``acc += x * x``, a shuffle tree and the warps; 64 threads of ``fmaf`` chains of a dot product), on rows of 1 to 4096 columns of every
    magnitude, signed zeros, NaNs, infinities and subnormals, and an rmsnorm of 128 threads: ``tests/test_gpu_reduction_order.py``. Every add is
    one rounding and never contracted, so with ``fma=False`` a product summed is rounded first (``acc + round(a * b)``): the product is one
    ``mul.rn.f32`` of inline PTX whatever ``fp_fusion`` says, which nothing packs or contracts with an add (note 2). The cost is the order's:
    a sequential row of 4096 columns is 4096 dependent adds. The ``reference`` mode computes the same order in torch (f32 adds, an fma as in note
    16, NaNs canonical): byte-identical to the binary.
18. **Float to integer casts.** ``x.to(i32, rounding=..., saturate=...)`` (also ``i64`` and ``u8``; from f32, or from bf16, f16, e4m3 or e5m2,
    which are widened to f32 first, exactly) says how it rounds and what a value out of range becomes; ``rounding`` has no default, and no other
    cast takes either. ``rounding="rn"`` rounds to the nearest integer, a tie to the even one (``2.5`` is 2, ``-0.5`` is 0), ``"rz"`` toward zero,
    ``"rd"`` down (toward ``-inf``), ``"ru"`` up; a subnormal is rounded as it is (``ru`` of ``1e-45`` is 1, ``rd`` of ``-1e-45`` is -1).
    ``saturate=True``, the default, clamps the rounded value to the type's range (an infinity to its bound) and makes a NaN 0, for every type
    alike: Rust's ``as`` with that rounding, CUDA's ``__float2int_rn`` family for 32 bits. ``saturate=False`` keeps the low ``bits`` bits of the
    rounded value as an ``i64`` (two's complement, modulo ``2**bits``), where a value outside the ``i64`` range has saturated first and a NaN is
    0: C's narrowing of an integer, and torch's cast of a float to ``u8``, which goes through ``int64`` (``-1.5`` to ``u8`` with ``rz`` is 255,
    ``300`` is 44). For ``i64`` the two modes are one. The default saturates because a saturated value is the representable one nearest to the
    rounded value, by one rule for every type, where a wrapped one is far from it and is asked for on purpose; the rounding has no default
    because a truncation and a rounding to nearest are both common, and differ on every value whose fraction is above one half.
    Each cast is a module-level helper of inline PTX (``_tf_f2i_s32_rn``), so the instruction is the contract and Triton's ``.to()`` (whose ``u8``
    is neither of these, note 1) is not involved: ``cvt.rni.s32.f32`` (``rzi``, ``rmi``, ``rpi`` for the others), which saturates and gives 0
    for a NaN, for a saturating ``i32``; the same then ``max.s32`` and ``min.s32`` with 0 and 255 for a saturating ``u8``, whose ``.to(tl.uint8)``
    then keeps the value; ``cvt.rni.s64.f32`` with a NaN made 0 (``testp.notanumber`` and ``selp``: the ``cvt`` makes a NaN ``-2**63``) for
    ``i64`` and for every wrapping cast, which then takes the low bits with ``.to(tl.int32)`` / ``.to(tl.uint8)`` (a truncation of an
    integer). None has ``.ftz``. The ``reference`` mode computes the declared mode exactly (in float64 and int64, not torch's cast), and is
    byte-identical to the binary on every one of the ``2**32`` f32 bit patterns for every type, rounding and mode (and on every bf16 and f16),
    measured on a B300 MIG slice (``tests/test_gpu_f2i.py``). Measured there, over every f32, against torch 2.9.1's CUDA casts:
    ``x.to(torch.int32)`` is ``rz`` saturating on every input (``cvt.rzi.s32.f32``); ``x.to(torch.int64)`` is ``rz`` except on the 16 777 214
    NaNs, which it makes ``-2**63``; ``x.to(torch.uint8)`` is ``rz`` wrapping on every input (c10 casts through ``int64``), and differs from
    ``rz`` saturating on 712 048 640; ``x.to(torch.int64).to(torch.int32)`` is ``rz`` wrapping; ``torch.round(x)``, ``torch.floor(x)`` and
    ``torch.ceil(x)`` then ``.to(torch.int32)`` are ``rn``, ``rd`` and ``ru`` saturating.
19. **Static slices** (the row template). ``x[k]``, ``x.select(0, k)`` with ``k`` a python int, of a read-only tensor parameter ``x`` of rank 2 or
    more, is read where it is: a parameter of the shape ``x.shape[1:]`` at the element offset ``k * prod(x.shape[1:])``, which a header line of
    the module states. It takes the role its shape has in the template: ``codes[k]`` of a ``[K, R]`` plane is an ``[R]`` operand,
    ``tl.load(codes + k * size_R + row)`` (the index of a ``gather_rows``, or any per-row value; ``[K, R, 1]`` gives ``[R, 1]``), ``bias[k]`` of a
    ``[K, D]`` parameter a ``[D]`` one, ``xs[k]`` of a ``[K, R, D]`` parameter an ``[R, D]`` one (read chunk by chunk in a chunked row), and
    ``tables[k]`` of a ``[K, N, D]`` parameter the table of a gather, ``tl.load(tables + k * D * size_N + i * D + cols, ...)``. A dynamic
    dimension of the offset is a runtime ``i32`` parameter of that dimension (``size_R``, after the template's own and the tables' rows), which a
    host derives from the shapes like any dimension scalar; a static offset is a number. The offset is ``i32``: ``x`` has a ``max_elements``
    precondition when its size is dynamic (or ``index-range`` holds it), so ``k * size_R + row`` is below its element count. A parameter read
    only through its slices is not loaded and not fitted to the template. Every other view is refused (rule ``view``): a select along another
    axis, of a computed value or of a parameter the kernel writes, an element of a rank-1 parameter (a ``DevScalar`` is that), an offset that is a
    product of two dynamic sizes; so is a returned slice (``view-value``: return ``x[k] + 0``), and the elementwise template takes none
    (``elementwise.slice``). The ``reference`` mode is torch's ``select``: a raw read, byte-identical (``tests/test_gpu_slices.py``).
20. **Fused regions** (``docs/rfcs/0001-plan-fusion.md``, :mod:`tilefish._fusion`). The arithmetic written in a ``@tilefish.graph`` body is
    planned as regions, each a kernel lowered here with ``fp_fusion=False`` and ``context_free=True``, so that a region is byte for byte the
    composition of one kernel per op (the unfused composition). Its code is the code of every kernel with ``fp_fusion=False`` (note 2), whose bits
    do not depend on the ops around them:

    * **every narrowing float cast** (to bf16 or f16, from f32, f16 or bf16) is one helper of inline PTX, ``cvt.rn.bf16.f32`` /
      ``cvt.rn.f16.f32`` (note 1): with ``.to()`` the folds of MLIR and LLVM keep an input NaN's payload through ``x.to(f32).to(bf16)`` in one
      kernel, and through a ``where`` between widened values narrowed after it (506 NaN outputs differed), where two kernels give the canonical
      NaN (``0x7FFF``) for each;
    * **a negation** gives the canonical NaN explicitly (note 16, as in every kernel);
    * **every float ``+``, ``-`` and ``*``** (f32, f16, bf16), and the last multiply of ``silu`` and the multiplies and add of ``gelu``, is one
      scalar instruction of inline PTX with ``.rn`` (note 2: ptxas 12.9 contracts a packed ``mul.rn.f32x2`` and ``add.rn.f32x2`` even with
      ``--fmad=false``).

    ``context_free`` adds the header line that says so, and lets a kernel take a **view where it is**: a ``Strided[...]`` parameter with a
    ``StrideOf`` per axis (a fused region's input that is a column block, a head or a transpose), which the elementwise template reads at
    ``sum(coordinate_k * stride_k)``, the coordinates as note 12 computes them (integer arithmetic, exact; a ``max_extent`` precondition holds the
    offsets within ``i32``). The row template takes none in revision 1 (rule ``row.strided``): a fused region of fusion version 1 then copies the
    view with ``tilefish.contiguous`` first; revision 2 reads it where it is (note 21).

    Every non-NaN value is the same either way, and the inline cast is ``.to()``'s bits on every f32 (``tests/test_gpu_fusion.py``).
21. **Revision 2 of the lowering** (``lower(..., lowering=2)``; the kernel's ``lowering`` attribute, which the regions of ``tilefish.fusion(version=2)``
    are made with, a kernel declared ``@tilefish.kernel(lowering=2)`` has, and every other kernel has as ``1``). The bytes of revision 1 but where
    revision 1 loses the sign of a ``-0.0`` scalar (the last bullet), other code, so other binaries and pins: a kernel's revision is frozen with what
    makes it (a hand-written kernel keeps revision 1 and its source unless it declares 2).

    * **Ordered sums by halving splits.** Revision 1 takes each step of a lane's chain with a ``tl.gather`` of the row (a pass through shared memory
      per step: 85 us for an rmsnorm of 4096 rows of 1024 columns on a B300 slice, where its three unfused kernels with a Triton tree sum take
      61). Revision 2 makes the columns at or past ``D`` ``-0.0`` (``fma(-0.0, 1.0, acc)`` for an fma sum), which ``add.rn`` without flush to zero
      leaves every value a chain can hold as it is (``+0.0 + -0.0`` is ``+0.0``, ``-0.0 + -0.0`` is ``-0.0``, a NaN a chain holds is already the
      canonical one), so a column past the end is skipped as revision 1 skips it; pads the row with such columns to whole rounds of ``lanes *
      vector`` (``tl.join``, data movement); reshapes it to ``[lanes, K]``, lane ``j``'s columns ``c * lanes * vector + j * vector + v`` in their order
      along the last axis; adds each lane's chain by a recursive helper that takes the low half of the columns, then the high half, each by a
      ``tl.split`` (data movement, exact) down to one column, one ``add.rn`` (``fma.rn``) each; then the tree over the lanes, each offset ``p[j] +
      p[j + o]`` for every lane at once with ``tl.gather`` (a lane whose partner is past the end is not read again: only live lanes reach lane 0), and
      lane 0's value by ``tl.reduce`` of (value, lane) pairs with a combine that keeps the lower lane (a select: any tree gives lane 0's bits). Every
      add is the one revision 1 makes, in its order: checked against the CUDA kernels of each order at every length from 1 to 4096 columns, on rows of
      every magnitude, signed zeros, NaNs, infinities and subnormals, and at every ``num_warps`` (``tests/test_gpu_reduction_order.py``). A
      sequential order (one lane), or a lane of more than ``SPLIT_MAX_STEPS = 1024`` steps at a bucket, keeps revision 1's helper. Measured (B300
      MIG slice, rows of 32 to 4096 and 512 to 16384 columns): 1.6 to 37 times faster than revision 1 (a lane of 512 steps 37 times), at the
      bandwidth of the row's bytes at 4096 columns.
    * **The warps of a row of ordered sums.** A row whose reductions are ordered sums or means of more than one lane (and order-free ones: ``max``,
      ``argmax``) has one warp up to ``BLOCK_D = 2048``, then one per 1024 columns (per 512 for an order of 512 lanes or more), at most 16: within 2%
      of the best of 1, 2, 4 and 8 warps over the measured grid (the template's rule, a warp per 256 columns, is up to 1.7 times slower). A row with a
      Triton tree reduction keeps the template's rule, which is its numerics (note 6); the warp count of a row of ordered sums is a speed choice.
    * **Views read where they are by the row template** (a context-free kernel): an ``[R, D]`` view at ``r * s0 + c * s1``, a ``[D]`` one at ``c * s0``,
      an ``[R]`` or ``[R, 1]`` one at ``r * s0``, the strides scalar arguments as in note 20.
    * **An fp8 cast of a context-free kernel** (or one with ``fp_fusion=False``) is inline PTX, ``cvt.rn.satfinite.e4m3x2.f32`` /
      ``.e5m2x2.f32`` of ``(+0.0, x)``, its low byte: the instruction ``.to()`` compiles to (note 1), out of the reach of the folds; ``.to()``'s bits
      on every f32 and every bf16 and f16 pattern (``tests/test_gpu_fusion_rows.py``).

    * **A scalar ``-0.0`` keeps its sign.** Triton 3.5.1 writes a literal ``-0.0`` (``tl.where(m, x, -0.0)``, ``tl.full(..., -0.0, ...)``, a
      ``-0.0`` argument of an inline-asm helper) as ``+0.0`` (measured on sm_103: ``0x00000000`` where ``0x80000000`` was written, with or without a
      ``+0.0`` in the same kernel), so revision 1 computes ``where(x > 0, x, -0.0)`` with ``+0.0`` (its fused and unfused plans alike, so fusion
      stays exact; against the ``reference`` mode, and the declared op, it is wrong). Revision 2 writes a ``-0.0`` scalar (f32, bf16, f16) by its
      bits, ``tl.full((), -2147483648, tl.int32).to(tl.float32, bitcast=True)``, and so does its padding of ordered sums. A ``const`` parameter of a
      hand-written kernel whose value is ``-0.0`` is a ``tl.constexpr``, which Triton still makes ``+0.0`` (a graph's region writes its value as a
      scalar, so it is covered). So a graph's version 1 and version 2 plans differ where a ``-0.0`` scalar meets a zero (also one that
      ``fp_contract="mul-add"`` makes: ``a * b - 0.0`` is ``fma(a, b, -0.0)``).

    ``mean(x, -1, order=...)`` (either revision) is the ordered sum times ``f32(1 / D)``, one ``mul.rn``: ``1.0 / D`` of the constexpr ``D``, a
    Python float that Triton rounds to f32, which is ``1 / D`` correctly rounded for every ``D`` up to ``2**20`` (the binary expansion of ``1 / D``
    has no run of more than ``log2(D)`` equal bits, so the double is never on a midpoint of f32), as the ``reference`` mode computes it. Not
    ``rcp.rn.f32(D)``: ptxas expands that next to the multiply into fused multiply-adds, which the contraction check refuses.

**``num_warps`` and the numerics identity** (decided on measurements, ``tests/test_gpu_reduction_order.py``). What depends on ``num_warps`` is
Triton's own reductions: ``tl.sum`` (``sum`` without ``order``, ``mean``) adds in a tree that the warp count shapes (note 6; on 1024 rows of
values of every magnitude, 669 to 878 rows differ between 1 warp and 2 to 16, at 96, 1000 and 4096 columns), and ``tl.max`` reduces in the same
tree. Per element ops, casts, ``argmax`` / ``argmin`` (a total order, note 8) and ordered sums (note 17) do not: their bytes are the same at every
warp count. ``num_warps`` is part of every kernel's identity all the same: recorded everywhere an identity is, the declaration digest, the compile
spec, the manifest entry (a replay refuses an entry whose ``num_warps`` is not the spec's) and the manifest digest. A binary of another warp
count is other machine code, and one rule for every kernel is checked mechanically where a per-kernel argument would not be. For a kernel with a
Triton reduction it is also its numerics: the same kernel with another warp count differs in the last bits, like a changed reduction order or
accumulator dtype (a numerics-changing rewrite in the README's terms). Pin it (``@tilefish.kernel(num_warps=N)``) where bytes must not move when a
size is added to a bucketed dimension, and treat a change of it, or of the rule that picks it (``clamp(BLOCK_D // 256, 1, 16)``), as a change of
the numerics contract; or give every reduction an ``order``, and the warp count is a choice of speed only.

The ``reference`` mode is held to the binary within tolerances that are measured per input distribution, not stated once:
a ULP count only means something where the result is far from zero, so for a bf16 output the bound is half a bf16 step of the
largest value of the reference (``2**-7 * max|ref|``) together with a fraction of elements allowed to differ, for the f32 softmax an
absolute error of 5e-7 (and ULP where meaningful), for the gelu an absolute error of 3e-5. They differ by dtype, kernel and
distribution (the mean of a layernorm's data, the scale of a softmax's; heavy tails, outliers, values of every magnitude) and by the dtype of
the output (f32, bf16, f16), and the measured bounds are tabulated, and asserted, in ``tests/test_gpu_reference.py``. Where the reference is exact
(IEEE math, notes 15 to 17) the binary is its bytes.
"""

from __future__ import annotations

import hashlib
import itertools
import keyword
import math
import pathlib
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ._specvalid import MAX_ELEMENTS
from .aot import CompileSpec, compile_specs
from .decorators import ExternOp, KernelOp
from .dims import Dim, DimExpr, as_dim, cdiv, dim_bounds, next_pow2
from .dtypes import BlockScaled
from .errors import TilefishLoweringError, TilefishTypeError
from .ir import Graph, Node, Scalar
from .signature import MISSING, Param, Signature
from .trace import Value

ELEMENTWISE_BLOCK = 1024
ELEMENTWISE_NUM_WARPS = 4
ROW_LANES_PER_WARP = 256
ROW_MAX_WARPS = 16
ARG_CHUNK = 1024  # an argmax / argmin row longer than this is walked in chunks of this many columns (note 8)
ARG_MAX_CHUNKS = 16  # ... when it has at most this many (unrolled); a longer row is one block
ARG_CHUNK_WARPS = 2
MAX_BLOCK = 1 << 20  # Triton's limit on the number of elements of a tensor
NUM_STAGES = 3
NUM_WARPS = (1, 2, 4, 8, 16, 32)
LOWERINGS = (1, 2)  # the revisions of the lowering (note 21): 1 is every kernel's, 2 the regions' of tilefish.fusion(version=2)
SPLIT_MAX_STEPS = 1024  # revision 2: an ordered sum takes its lanes' chains by halving splits when each lane has at most this many steps (note 21)
ORDERED_ONE_WARP = 2048  # revision 2: a row of ordered sums of BLOCK_D up to this has one warp (note 21) ...
ORDERED_LANES_PER_WARP = 1024  # ... a longer one a warp per this many columns ...
ORDERED_WIDE_LANES = 512  # ... or per half as many when an order has this many lanes or more
DEFAULT_MODULE = "tilefish_generated"
TEMPLATES = ("elementwise", "row")
INDEX_MAX = MAX_ELEMENTS  # offsets are i32: a tensor has at most 2**31 - 1 elements
# Names a kernel parameter cannot keep in the generated function: what the module itself uses, and what
# Triton's JIT launcher takes as launch options (a parameter of that name could not be passed by keyword).
_RESERVED = frozenset({"tl", "triton", "float", "libdevice", "grid", "warmup", "debug", "num_warps", "num_stages", "num_ctas", "maxnreg"})

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")  # ASCII only, like the identifiers a compile spec accepts
_MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*\Z")

# the Triton spelling of each dtype the lowering handles
_TL = {
    "bf16": "bfloat16",
    "f16": "float16",
    "f32": "float32",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
    "e4m3": "float8e4nv",
    "e5m2": "float8e5",
    "bool": "int1",  # a mask: a value inside the body, never loaded or stored
}
_ARITHMETIC = frozenset({"f32", "bf16", "f16", "i32", "i64"})
_MATH = frozenset({"f32"})
_BINARY = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_COMPARE = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_LOGICAL = {"and": "&", "or": "|", "xor": "^"}
_UNARY = ("rsqrt", "exp", "sqrt", "silu", "gelu", "sin", "cos", "tanh", "erf", "sin_approx", "rcp")
_NEG = frozenset({"f32", "bf16", "f16", "i32", "i64"})
_RN = frozenset({"f32", "bf16", "f16"})  # the float dtypes whose + - * fp_fusion=False writes as one inline instruction each (note 2)  # neg: the floats by CUDA's neg (note 16), the signed integers by 0 - x
_REDUCTIONS = ("sum", "mean", "max")
_ARG_REDUCTIONS = ("argmax", "argmin")
_OTHER_OPS = ("not", "where", "iota", "gather_rows", "scatter_rows", "floordiv", "mod")  # dtypes checked like a load's
# Triton's own type of a python literal: a scalar of any other dtype is written as a typed constant (tl.cast(0.1, tl.bfloat16)),
# because Triton compares a bf16 or f16 tensor with a float literal in f32 (the literal is not rounded to the tensor's dtype first)
_LITERAL_DTYPES = frozenset({"f32", "i32"})
_INDEX_SENTINEL = 2147483647  # the column of a masked lane in an argmax / argmin: it loses every tie

_FULL, _COL, _ROW = "full", "col", "row"  # row template: [R, D], [D], [R] / [R, 1]
_SCALAR, _DEV = "scalar", "devscalar"  # rank 0: a value read from a DevScalar (broadcast), and the DevScalar parameter itself (a pointer to one element)


# -- f32 math in IEEE mode (note 13) ----------------------------------------------------------------
# The PTX that CUDA C++ compiles each to without fast math and without flush to zero (nvcc's defaults: -ftz=false -prec-div=true -prec-sqrt=true,
# what torch's CUDA kernels are built with), read from NVRTC 12.8 for sm_100, written out as inline PTX so that Triton, which links libdevice with
# __CUDA_FTZ = 1 and emits ``.ftz`` for its own division and square root, compiles exactly this. ``add`` and ``mul`` carry ``.rn``, which rounds as the
# plain form does and keeps ptxas from contracting them with an instruction outside the block.

_EXPF = (  # expf(x): a range reduction j = round(x * log2 e) (fma.rm onto 1.5 * 2**23 + 1), r = x * log2 e - j in two parts, 2**r * 2**j
    "fma.rn.f32 t0, $1, 0f3BBB989D, 0f3F000000;",  # x * log2(e) / 252 + 0.5
    "cvt.sat.f32.f32 t1, t0;",  # clamped to [0, 1]
    "fma.rm.f32 t2, t1, 0f437C0000, 0f4B400001;",  # * 252 + 12582913, rounded down: j in the low bits
    "add.rn.f32 t3, t2, 0fCB40007F;",  # - 12583039
    "neg.f32 t3, t3;",
    "fma.rn.f32 t4, $1, 0f3FB8AA3B, t3;",  # x * log2(e) (high part) - j
    "fma.rn.f32 t4, $1, 0f32A57060, t4;",  # + x * log2(e) (low part)
    "mov.b32 r0, t2;",
    "shl.b32 r1, r0, 23;",  # 2**j as the bits of an f32
    "mov.b32 t5, r1;",
    "ex2.approx.ftz.f32 t6, t4;",  # 2**r (r in [-1, 1]: nothing to flush)
)
_SILU = (  # x / (1 + expf(-x)), as torch's silu kernel compiles: the negation folded into the constants, 2**r * 2**j + 1 one fma
    "fma.rn.f32 t0, $1, 0fBBBB989D, 0f3F000000;",
    "cvt.sat.f32.f32 t1, t0;",
    "fma.rm.f32 t2, t1, 0f437C0000, 0f4B400001;",
    "add.rn.f32 t3, t2, 0fCB40007F;",
    "neg.f32 t3, t3;",
    "fma.rn.f32 t4, $1, 0fBFB8AA3B, t3;",
    "fma.rn.f32 t4, $1, 0fB2A57060, t4;",
    "mov.b32 r0, t2;",
    "shl.b32 r1, r0, 23;",
    "mov.b32 t5, r1;",
    "ex2.approx.ftz.f32 t6, t4;",
    "fma.rn.f32 t7, t6, t5, 0f3F800000;",  # expf(-x) + 1, one rounding (nvcc contracts the multiply of expf with the add)
    "div.rn.f32 $0, $1, t7;",
)


_IEEE_HEADER = "math: ieee (note 13: / and sqrt are IEEE 754, exp is expf, silu torch's, rsqrt rsqrtf, sin and tanh keep a subnormal; no flush to zero)"


def _asm(instructions: "Iterable[str]", constraints: str, args: str, regs: str = "", dtype: str = "float32") -> "list[str]":
    """The body of a helper that is one block of inline PTX (one element per instance), its instructions one per line of the string."""
    lines = ["return tl.inline_asm_elementwise("]
    if regs:
        lines.append(f'    "{{ {regs} "')
        lines += [f'    "{i} "' for i in instructions]
        lines.append('    "}",')
    else:
        (only,) = instructions
        lines.append(f'    "{only}",')
    lines += [f'    "{constraints}",', f"    [{args}],", f"    dtype=tl.{dtype},", "    is_pure=True,", "    pack=1,", ")"]
    return lines


_ERFF = (  # CUDA's erff (nvcc's defaults, no flush to zero) as NVRTC 12.8 to 13.1 emit it, its one branch made a select (note 15)
    "abs.f32 t0, $1;",
    "setp.ltu.f32 p0, t0, 0f3F8060FE;",  # |x| < 1.0029547 (or a NaN): erf is the polynomial below
    "setp.ge.f32 p1, t0, 0f3F8060FE;",
    "mul.rn.f32 t1, $1, $1;",
    "selp.f32 t2, t0, t1, p1;",  # the variable of the polynomial: |x| above the threshold, x * x below
    "selp.f32 t3, 0f38EB4C3A, 0f38B1E96A, p1;",
    "selp.f32 t4, 0fBAAE005B, 0fBA574D20, p1;",
    "fma.rn.f32 t3, t3, t2, t4;",
    "selp.f32 t4, 0f3C09919F, 0f3BAAD5EA, p1;",
    "fma.rn.f32 t3, t3, t2, t4;",
    "selp.f32 t4, 0fBD24D99A, 0fBCDC1BE7, p1;",
    "fma.rn.f32 t3, t3, t2, t4;",
    "selp.f32 t4, 0f3E235519, 0f3DE718AF, p1;",
    "fma.rn.f32 t3, t3, t2, t4;",
    "selp.f32 t4, 0f3F69B4F9, 0fBEC093AC, p1;",
    "fma.rn.f32 t3, t3, t2, t4;",
    "selp.f32 t4, 0f3F210A14, 0f3E0375D3, p1;",
    "fma.rn.f32 t3, t3, t2, t4;",
    "neg.f32 t5, t0;",
    "selp.f32 t5, t5, $1, p1;",  # -|x| above, x below
    "fma.rn.f32 t6, t3, t5, t5;",  # p * s + s: erf(x) below the threshold
    "ex2.approx.ftz.f32 t7, t6;",
    "sub.rn.f32 t7, 0f3F800000, t7;",  # 1 - 2**(p * s + s)
    "copysign.f32 t7, $1, t7;",  # with the sign of x: erf(x) above the threshold
    "selp.f32 $0, t6, t7, p0;",
)


_IEEE: "dict[str, tuple[str, tuple[str, ...], list[str]]]" = {  # op -> (parameters, comment, body) of its helper in IEEE math (note 13)
    "div": ("a, b", ("IEEE 754 division: correctly rounded, subnormals kept (CUDA's '/', not Triton's div.full or div.rn.ftz)",),
            _asm(["div.rn.f32 $0, $1, $2;"], "=f,f,f", "a, b")),
    "sqrt": ("x", ("IEEE 754 square root: correctly rounded, a subnormal argument kept (CUDA's sqrtf, not sqrt.rn.ftz)",),
             _asm(["sqrt.rn.f32 $0, $1;"], "=f,f", "x")),
    "rsqrt": ("x", ("CUDA's rsqrtf: rsqrt.approx without flush to zero",), _asm(["rsqrt.approx.f32 $0, $1;"], "=f,f", "x")),
    "exp": ("x", ("CUDA's expf without flush to zero (note 13 of tilefish.lower): the sequence nvcc emits",),
            _asm([*_EXPF, "mul.rn.f32 $0, t6, t5;"], "=f,f", "x", ".reg .f32 t<7>; .reg .b32 r<2>;")),
    "silu": ("x", ("torch's silu, x / (1 + expf(-x)), as nvcc compiles it without flush to zero (note 13 of tilefish.lower)",),
             _asm(list(_SILU), "=f,f", "x", ".reg .f32 t<8>; .reg .b32 r<2>;")),
    "sin": ("x", ("libdevice's sinf (built with flush to zero), but a subnormal argument is its own sine, as without flush to zero",),
            ["return tl.where(tl.abs(x) < 1.1754943508222875e-38, x, tl.sin(x))"]),
    "tanh": ("x", ("libdevice's tanhf (built with flush to zero), but a subnormal argument is its own tanh, as without flush to zero (note 14)",),
             ["return tl.where(tl.abs(x) < 1.1754943508222875e-38, x, libdevice.tanh(x))"]),
    "erf": ("x", ("CUDA's erff without flush to zero, the sequence nvcc emits (CUDA 12.8 to 13.1; note 15 of tilefish.lower)",),
            _asm(list(_ERFF), "=f,f", "x", ".reg .f32 t<8>; .reg .pred p<2>;")),
}  # (cos is libdevice's cosf in both: it has the same bits with and without flush to zero, note 10)

# Ops that are one PTX instruction in either math (note 16), and the adds of an ordered sum (note 17): (parameters, comment, body) of the helper
_PTX: "dict[str, tuple[str, tuple[str, ...], list[str]]]" = {
    "sin_approx": ("x", ("sin.approx.f32: CUDA's __sinf (note 16 of tilefish.lower)",), _asm(["sin.approx.f32 $0, $1;"], "=f,f", "x")),
    "rcp": ("x", ("rcp.rn.f32: 1 / x correctly rounded, subnormals kept (CUDA's __frcp_rn; note 16 of tilefish.lower)",),
            _asm(["rcp.rn.f32 $0, $1;"], "=f,f", "x")),
    "fma": ("a, b, c", ("fma.rn.f32: a * b + c rounded once, whatever fp_fusion says (CUDA's fmaf; note 16 of tilefish.lower)",),
            _asm(["fma.rn.f32 $0, $1, $2, $3;"], "=f,f,f,f", "a, b, c")),
    "add.rn": ("a, b", ("add.rn.f32: one rounding, never contracted with a multiply (an ordered sum, note 17 of tilefish.lower)",),
               _asm(["add.rn.f32 $0, $1, $2;"], "=f,f,f", "a, b")),
    # separate roundings (notes 2, 17 and 20): one scalar instruction each, which no compiler packs into f32x2 and contracts
    **{
        f"{op}.rn.{dt}": (
            "a, b",
            (f"{op}.rn.{ptx}: one rounding, never packed or contracted with a neighbour (note 2 of tilefish.lower)",),
            _asm([f"{op}.rn.{ptx} $0, $1, $2;"], f"={reg},{reg},{reg}", "a, b", dtype=tl_name),
        )
        for op in ("add", "sub", "mul")
        for dt, ptx, reg, tl_name in (("f32", "f32", "f", "float32"), ("f16", "f16", "h", "float16"), ("bf16", "bf16", "h", "bfloat16"))
    },
    "cvt.bf16": ("x", ("cvt.rn.bf16.f32: f32 to bf16, to nearest even, every NaN the canonical 0x7FFF (note 1); inline, out of the reach of folds",),
                 _asm(["cvt.rn.bf16.f32 $0, $1;"], "=h,f", "x", dtype="bfloat16")),
    "cvt.f16": ("x", ("cvt.rn.f16.f32: f32 to f16, to nearest even, every NaN the canonical 0x7FFF (note 1); inline, out of the reach of folds",),
                _asm(["cvt.rn.f16.f32 $0, $1;"], "=h,f", "x", dtype="float16")),
    # revision 2 (note 21): a narrowing cast to fp8 of a context-free kernel, the instruction .to() compiles to (note 1), its result in the low byte
    **{
        f"cvt.{dt}": (
            "x",
            (f"cvt.rn.satfinite.{dt}x2.f32 of (+0.0, x): x in the low byte, to nearest even, saturated at the largest finite value, NaN 0x7F (note 1); "
             "inline, out of the reach of folds (note 21)",),
            _asm([f"cvt.rn.satfinite.{dt}x2.f32 $0, 0f00000000, $1;"], "=h,f", "x", dtype="uint16"),
        )
        for dt in ("e4m3", "e5m2")
    },
    # a negation whose NaN is the canonical one whatever ptxas folds it into (note 16): ptxas may make neg a negated operand of its consumer
    "negc.f32": ("x", ("-x: neg.f32, and the canonical NaN 0x7FFFFFFF for a NaN, explicitly (note 16; a folded neg.f32 flips a NaN's sign)",),
                 _asm(["testp.notanumber.f32 p, $1;", "neg.f32 t, $1;", "selp.f32 $0, 0f7FFFFFFF, t, p;"], "=f,f", "x", ".reg .pred p; .reg .f32 t;")),
    "negc.f16": ("x", ("-x: neg.f16, and the canonical NaN 0x7FFF for a NaN, explicitly (note 16)",),
                 _asm(["setp.nan.f16 p, $1, $1;", "neg.f16 t, $1;", "selp.b16 $0, 0x7FFF, t, p;"], "=h,h", "x", ".reg .pred p; .reg .b16 t;", dtype="float16")),
    "negc.bf16": ("x", ("-x: neg.bf16, and the canonical NaN 0x7FFF for a NaN, explicitly (note 16)",),
                  _asm(["setp.nan.bf16 p, $1, $1;", "neg.bf16 t, $1;", "selp.b16 $0, 0x7FFF, t, p;"], "=h,h", "x", ".reg .pred p; .reg .b16 t;", dtype="bfloat16")),
}


# -- float to integer casts (note 18) ------------------------------------------------------------------
# Each is a block of inline PTX, so that the instruction is the contract (not what Triton's .to() lowers to). The f32 source has no
# ``.ftz``: a subnormal rounds up to 1 (``ru``) or down to -1 (``rd``) as it should.

_F2I_ROUND = {"rn": "rni", "rz": "rzi", "rd": "rmi", "ru": "rpi"}  # the integer rounding of the cvt of each mode


def _f2i_helper(kind: str, rounding: str) -> "tuple[tuple[str, ...], list[str]]":
    """The comment and the body of the helper that converts an f32 to ``kind`` with ``rounding``: ``s32`` (saturating, NaN 0), ``u8`` (an
    ``i32`` in ``[0, 255]``: saturating, NaN 0) or ``s64`` (saturating, NaN 0; the wrapping casts take their low bits)."""
    r = _F2I_ROUND[rounding]
    if kind == "s32":  # PTX: a float to integer cvt saturates, and a NaN of an f32 is 0 for a destination narrower than 64 bits
        return (f"f32 to i32, {rounding}: cvt.{r}.s32.f32, which saturates (an infinity to the bound) and gives 0 for a NaN",), _asm(
            [f"cvt.{r}.s32.f32 $0, $1;"], "=r,f", "x", dtype="int32"
        )
    if kind == "u8":
        return (f"f32 to u8, {rounding}, saturating: cvt.{r}.s32.f32 (NaN 0), then clamped to [0, 255]",), _asm(
            [f"cvt.{r}.s32.f32 t, $1;", "max.s32 t, t, 0;", "min.s32 $0, t, 255;"], "=r,f", "x", ".reg .s32 t;", dtype="int32"
        )
    assert kind == "s64", kind
    return (f"f32 to i64, {rounding}: cvt.{r}.s64.f32, which saturates; a NaN, which the cvt makes -2**63, is 0",), _asm(
        [f"cvt.{r}.s64.f32 t, $1;", "testp.notanumber.f32 p, $1;", "selp.s64 $0, 0, t, p;"], "=l,f", "x", ".reg .s64 t; .reg .pred p;", dtype="int64"
    )


# -- the result ---------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class LoweredKernel:
    """The generated Triton source of one kernel, and a typed stub describing its parameters.

    ``extern`` is an :class:`~tilefish.decorators.ExternOp` for the generated function: pass it to
    :func:`tilefish.aot.compile_specs` to get its compile specs (one per bucket of a bucketed
    ``D``). Its triton reference is ``module:fn_name``. ``kernel`` is the ``@tilefish.kernel`` it was
    lowered from, which the ``reference`` mode of :func:`tilefish.run` interprets.
    """

    name: str
    fn_name: str
    module: str
    template: str  # "row" | "elementwise"
    source: str
    extern: ExternOp
    num_warps: int
    kernel: "KernelOp | None" = None
    # what each line of ``source`` was written for (index 0 is line 1): ``None`` (the module's own code), or ``{"node": <index of the node in the
    # kernel's graph>, "op": ..., "loc": "file:line" of the body}`` (``"origin"`` for a fused region's: the op's position in the region's body), or
    # ``{"what": "load x"}`` for code of no op. For the traceability views (tilefish.explain); not in the source, so no digest depends on it.
    line_map: "tuple[Any, ...]" = field(default=(), repr=False, compare=False)

    def run(self, /, *args: Any, mode: str = "jit", **kwargs: Any) -> Any:
        """``tilefish.run(self, *args, mode=mode, **kwargs)``."""
        from .runtime import run

        return run(self, *args, mode=mode, **kwargs)

    @property
    def fp_fusion(self) -> bool:
        """Whether the specs ask Triton for FMA contraction (``enable_fp_fusion``)."""
        return bool(self.extern.fp_fusion)

    @property
    def source_sha256(self) -> str:
        """The sha256 of :attr:`source`: it is in the declaration digest and in every spec, and a manifest records it."""
        return str(self.extern.source_sha256)

    @property
    def bucket_num_warps(self) -> "dict[str, int]":
        """The warp count of each bucket by stem suffix (``{".D96": 1, ".D1000": 4}``); empty when ``num_warps`` is
        one value for every binary (an unbucketed kernel, or one that pins it)."""
        return dict(self.extern.bucket_num_warps)

    @property
    def ref(self) -> str:
        """``"module:function"``, the Triton reference of the generated kernel."""
        return f"{self.module}:{self.fn_name}"

    @property
    def grid(self) -> str:
        """The launch grid as an expression over dimension names, e.g. ``(M,)``."""
        return self.extern.grid_string or ""

    @property
    def filename(self) -> str:
        """The module's path relative to a source root: ``tilefish_generated/rmsnorm.py``."""
        return self.module.replace(".", "/") + ".py"

    def specs(self) -> "list[CompileSpec]":
        """``compile_specs(self.extern)``."""
        return compile_specs(self.extern)

    def write(self, directory: "str | pathlib.Path") -> pathlib.Path:
        """Write the source to ``<directory>/<module path>.py`` (creating folders) and return the path.

        With ``directory`` on ``sys.path`` the module imports as ``self.module``.
        """
        path = pathlib.Path(directory) / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.source.encode("utf-8"))
        return path

    def __repr__(self) -> str:
        return f"<LoweredKernel {self.name}: {self.template} template, {self.ref}>"


# -- errors -------------------------------------------------------------------------------------


def _fail(kernel: str, template: "str | None", rule: str, detail: str) -> TilefishLoweringError:
    where = f"kernel '{kernel}'" + (f" with the {template} template" if template else "")
    return TilefishLoweringError(f"cannot lower {where}: rule '{rule}' failed: {detail}", rule=rule)


def _label(v: Value) -> str:
    return f"parameter '{v.name}' ({v.type})" if v.is_param else f"{v.ref} ({v.type})"


def _all_values(graph: Graph) -> "list[Value]":
    """Every tensor of the kernel once, in IR order: parameters, then node outputs, then outputs."""
    seen: dict[Value, None] = {}
    for v in graph.inputs:
        seen.setdefault(v, None)
    for n in graph.nodes:
        for v in n.outputs:
            seen.setdefault(v, None)
    for v in graph.outputs:
        seen.setdefault(v, None)
    return list(seen)


def _is_one(d: DimExpr) -> bool:
    return d.is_static and d.value == 1


def _shape_text(shape: "Iterable[DimExpr]") -> str:
    return "[" + ", ".join(str(d) for d in shape) + "]"


# -- how many elements a tensor can have ---------------------------------------------------------------


_bounds = dim_bounds  # the interval of a dimension expression (tilefish.dims)


def _numel_bounds(shape: "Iterable[DimExpr]") -> "tuple[int, int | None]":
    """Fewest and most elements a tensor of this shape can have (``None``: unbounded)."""
    lo = 1
    hi: "int | None" = 1
    for d in shape:
        d_lo, d_hi = _bounds(d)
        lo *= d_lo
        hi = None if hi is None or d_hi is None else hi * d_hi
    return lo, hi


def _check_index_range(name: str, values: "list[Value]") -> None:
    """Reject a tensor that has more elements than i32 offsets reach, whatever the dynamic sizes are."""
    for v in values:
        lo, hi = _numel_bounds(v.type.shape)
        if lo > INDEX_MAX or (hi is not None and hi > INDEX_MAX):
            what = f"{lo} elements" if lo == hi else f"at least {lo} elements" if lo > INDEX_MAX else f"up to {hi} elements (its largest bucket)"
            raise _fail(
                name,
                None,
                "index-range",
                f"{_label(v)} has {what}, more than the {INDEX_MAX} that the i32 offsets of a generated kernel reach",
            )


# -- checks that do not depend on the template --------------------------------------------------


def _check_dtype(kernel: str, what: str, dtype: Any) -> None:
    if isinstance(dtype, BlockScaled):
        raise _fail(kernel, None, "dtype", f"{what} has block-scaled dtype {dtype!r}; block-scaled tensors are not lowered")
    if dtype.name not in _TL:
        raise _fail(
            kernel,
            None,
            "dtype",
            f"{what} has dtype {dtype}, which has no plain Triton element type here "
            f"(supported: {', '.join(_TL)}); packed sub-byte formats and scales are not lowered",
        )


def _check_common(op: KernelOp) -> None:
    """Rules every template shares: parameters, supported ops, and the dtypes each op is defined for."""
    graph, name = op.graph, op.name
    for p in op.signature.params:
        if p.kind == "const" and p.default is MISSING:
            raise _fail(
                name, None, "const-default", f"const parameter '{p.name}' has no default; the value is compiled in as a constexpr"
            )
    strided = _strided_params(op)
    for v in graph.inputs:
        if not v.type.contiguous and v.name not in strided:
            raise _fail(
                name,
                None,
                "view-input",
                f"parameter '{v.name}' is a View ({v.type}); strided views are not lowered, pass a contiguous tensor",
            )
        _check_dtype(name, f"parameter '{v.name}'", v.type.dtype)
    for n in graph.nodes:
        if "kind" in n.attrs:
            raise _fail(
                name,
                None,
                "declared-op-call",
                f"the body calls the {n.attrs['kind']} op '{n.op}'; only the built-in ops (+ - * /, comparisons, & | ^ ~ of masks, "
                "where, iota, // and % by a constant, silu, gelu, rsqrt, exp, sqrt, sin, cos, tanh, erf, sin_approx, rcp, neg, fma, .to, sum, mean, max, "
                "argmax, argmin, gather_rows, scatter_rows, load and stores) are lowered",
            )
        if n.op == "split":
            views = [str(t) for t in n.output_types if not t.contiguous]
            note = f" (its chunks are views: {views[0]})" if views else ""
            raise _fail(
                name,
                None,
                "split",
                f"split is not lowered{note}: a lowered kernel takes and gives contiguous tensors; a strided view goes to an "
                "extern with a Strided[...] parameter",
            )
        why = _static_slice_problem(n) if n.op == "select" else ""
        if n.op in ("reshape", "permute") or why:
            slice_note = (
                f" (the row template reads x[k] of a read-only tensor parameter, a static slice (note 19); this select is not one: {why})"
                if why
                else ""
            )
            raise _fail(
                name,
                None,
                "view",
                f"{n} is a view operation, and strided views are not lowered: a lowered kernel takes and gives contiguous tensors{slice_note}. "
                "Write the kernel over the tensor the view is of, or declare an extern with a Strided[...] parameter",
            )
        if n.op == "copy":
            raise _fail(
                name,
                None,
                "copy",
                f"{n} is tilefish.copy, a copy step of a graph's plan: in a kernel body a value is already its own (use it as it is)",
            )
        for v in n.outputs:
            if not v.type.contiguous and n.op != "select":  # a static slice is read at its offset (note 19)
                raise _fail(name, None, "view-value", f"{_label(v)} is a View; strided views are not lowered")
        _check_node_dtypes(name, n)
    slices = {n.outputs[0] for n in graph.nodes if n.op == "select"}
    for v in graph.outputs:
        if v in slices:
            raise _fail(
                name, None, "view-value", f"the returned {_label(v)} is a slice of a parameter; return what is computed from it (a copy is x[k] + 0)"
            )
        if not v.type.contiguous:
            raise _fail(name, None, "view-value", f"the returned {_label(v)} is a View; strided views are not lowered")
    _check_index_range(name, _all_values(graph))


def _strided_params(op: KernelOp) -> "dict[str, list[str]]":
    """The ``Strided[...]`` parameters of a context-free kernel (a fused region that reads a view where it is, note 20), each with the names of
    its stride parameters by axis; empty for every other kernel, whose parameters are contiguous."""
    if not getattr(op, "context_free", False):
        return {}
    strides: "dict[tuple[str, int], str]" = {}
    for p in op.signature.params:
        for arg, axis in p.stride_of or ():
            strides[(arg, axis)] = p.name
    out: "dict[str, list[str]]" = {}
    for p in op.signature.params:
        if p.kind == "tensor" and p.strided:
            assert p.type is not None
            out[p.name] = [strides[(p.name, axis)] for axis in range(p.type.rank)]
    return out


def _static_slice_problem(n: Node) -> str:
    """Why the select ``n`` is not a static slice that the row template reads (note 19), or ``""`` when it is one: ``x[k]`` (``x.select(0, k)``,
    ``k`` a python int) of a read-only tensor parameter of rank 2 or more, whose element offset ``k * prod(x.shape[1:])`` is affine in plain
    dimensions (each a runtime ``i32`` scalar of the kernel, or a number)."""
    x, out = n.inputs[0], n.outputs[0]
    if n.attrs["axis"] != 0:
        return f"it selects along axis {n.attrs['axis']}, and a slice is along the first axis"
    if not (isinstance(x, Value) and x.is_param):
        return f"{_label(x) if isinstance(x, Value) else x} is not a parameter of the kernel, so it has no place in memory to read"  # type: ignore[arg-type]
    if x.access != "in":
        return f"the {x.access} parameter '{x.name}' is written by the kernel"
    if x.type.rank < 2:
        return f"it is one element of the rank-1 parameter '{x.name}' (a DevScalar is how a kernel reads one element)"
    offset = out.type.offset
    if offset is None or any(not isinstance(a, Dim) for a, _ in offset.folded().terms):
        return f"its offset {offset} is not affine in the dimensions (a product of two dynamic sizes)"
    return ""


def _check_node_dtypes(kernel: str, n: Node) -> None:
    if n.op in _BINARY:
        dt = n.outputs[0].type.dtype
        if n.op == "div":
            if dt.name != "f32":
                raise _fail(
                    kernel,
                    None,
                    "division-dtype",
                    f"{n}: division of {dt} tensors is not lowered; Triton divides f16/bf16 in f32 and returns f32, "
                    "an implicit promotion. Cast the operands with .to(f32) first",
                )
        elif dt.name not in _ARITHMETIC:
            raise _fail(
                kernel,
                None,
                "arithmetic-dtype",
                f"{n}: arithmetic on {dt} is not lowered (supported: {', '.join(sorted(_ARITHMETIC))}); cast with .to() first",
            )
    elif n.op in _UNARY or n.op == "fma":
        dt = n.outputs[0].type.dtype
        if dt.name not in _MATH:
            raise _fail(
                kernel,
                None,
                "math-dtype",
                f"{n}: {n.op} of a {dt} tensor is not lowered (Triton's math functions are f32 here); cast with .to(f32) first",
            )
    elif n.op == "neg":
        dt = n.outputs[0].type.dtype
        if dt.name not in _NEG:
            raise _fail(
                kernel,
                None,
                "arithmetic-dtype",
                f"{n}: negation of a {dt} tensor is not lowered (supported: {', '.join(sorted(_NEG))}); cast with .to() first",
            )
    elif n.op in _REDUCTIONS:
        dt = n.inputs[0].type.dtype  # type: ignore[union-attr]
        if dt.name != "f32":
            raise _fail(
                kernel,
                None,
                "reduction-dtype",
                f"{n}: {n.op} of a {dt} tensor is not lowered (Triton would accumulate in {dt}); cast with .to(f32) first",
            )
    elif n.op == "cast":
        src = n.inputs[0].type.dtype  # type: ignore[union-attr]
        dst = n.outputs[0].type.dtype
        _check_dtype(kernel, f"the source of {n}", src)
        _check_dtype(kernel, f"the target of {n}", dst)
        if src.is_float and dst.is_int and n.attrs.get("round") not in _F2I_ROUND:
            raise _fail(
                kernel,
                None,
                "cast.float-to-int",
                f"{n}: casting {src} to {dst} without a rounding mode is not lowered: a float to integer cast says it "
                "(x.to(i32, rounding='rn' | 'rz' | 'rd' | 'ru', saturate=True | False); note 18 of the numerics contract)",
            )
    elif n.op == "load":
        _check_dtype(kernel, f"the DevScalar read by {n}", n.outputs[0].type.dtype)
    elif n.op == "select":  # a static slice (note 19): its dtype is the parameter's, checked with the parameters
        pass
    elif n.op in _COMPARE or n.op in _LOGICAL or n.op in _OTHER_OPS or n.op in _ARG_REDUCTIONS:
        for v in (*n.outputs, *(i for i in n.inputs if isinstance(i, Value))):
            _check_dtype(kernel, f"{v.ref} of {n}", v.type.dtype)
    elif n.op != "store":
        raise _fail(kernel, None, "unsupported-op", f"the op '{n.op}' is not lowered")


# -- tables: the parameters a gather reads or a scatter writes at computed rows ---------------------------------------------------


@dataclass
class _Tables:
    """The parameters that ``gather_rows`` reads and ``scatter_rows`` writes at computed rows (note 9), and the static slices (note 19).

    ``params`` maps each to ``"gather"`` or ``"scatter"``, in the order the kernel declares them (then the slices that are gathered from, in IR
    order); ``only`` are those that nothing else in the body reads (the prologue does not load them and the template does not fit them to
    its shapes: their rows are addressed by the index); ``versions`` are the versions of the targets that the scatters make (written in place,
    read by nothing). ``slices`` maps the value of each static slice ``x[k]`` to its select, and ``planes`` are the parameters that the body
    reads through static slices only (not loaded, not fitted: each slice is read at its own offset).
    """

    params: "dict[Value, str]"
    only: "set[Value]"
    versions: "set[Value]"
    slices: "dict[Value, Node]"
    planes: "set[Value]"

    def skip(self, v: Value) -> bool:
        """Whether the template leaves ``v`` out of its shapes: a table that only the index addresses, the version a scatter makes, or a
        parameter read through static slices only."""
        return v in self.only or v in self.versions or v in self.planes


def _tables(op: KernelOp) -> _Tables:
    graph, name = op.graph, op.name
    found: dict[Value, str] = {}
    versions: set[Value] = set()
    slices = {n.outputs[0]: n for n in graph.nodes if n.op == "select"}  # every select left is a static slice (_check_common)
    for n in graph.nodes:
        if n.op == "gather_rows":
            x = n.inputs[0]
            assert isinstance(x, Value)
            if x in slices:  # a table at a static offset of a parameter: tables[k] of a [K, N, D] parameter (note 19)
                found.setdefault(x, "gather")
                continue
            if not (x.is_param and x.access == "in") or x.type.devscalar:
                raise _fail(
                    name,
                    None,
                    "gather.table",
                    f"{n}: the table {_label(x)} is not a read-only tensor parameter of the kernel. A gather reads rows of a tensor in memory "
                    "at indices computed in the kernel, so its table is a Tensor[...] parameter: a computed value has no rows in memory, and "
                    "a parameter the kernel writes would race with the reads of the other row programs",
                )
            found.setdefault(x, "gather")
        elif n.op == "scatter_rows":
            t = n.inputs[0]
            assert isinstance(t, Value)
            if not (t.is_param and t.access == "inout") or t in found:
                raise _fail(
                    name,
                    None,
                    "scatter.target",
                    f"{n}: the target '{n.attrs['target']}' is written before this scatter (by a store or another scatter). The rows of a "
                    "scatter target are written at indices computed in every program, so one scatter is its only write",
                )
            found[t] = "scatter"
            versions.add(n.outputs[0])
    for t in found:
        if t.type.rank > 2:
            raise _fail(
                name, None, "table-rank", f"{_label(t)} has rank {t.type.rank}; a gather or a scatter addresses rows of a [N, D] or [N] tensor"
            )
    used: set[Value] = set()
    for n in graph.nodes:
        ins = n.inputs[1:] if n.op in ("gather_rows", "scatter_rows") else () if n.op == "select" else n.inputs  # a slice reads at its offset
        used.update(i for i in ins if isinstance(i, Value))
    used.update(graph.outputs)
    for t, kind in found.items():
        if kind == "scatter" and (t in used or used & versions):
            raise _fail(
                name,
                None,
                "scatter.target",
                f"the scatter target {_label(t)} is also read, stored into or returned by the kernel. Its rows are written at indices "
                "computed in every program, so any other access races with them: a scatter is the only use of its target",
            )
    ordered = {v: found[v] for v in graph.inputs if v in found}
    ordered.update({v: found[v] for v in slices if v in found})  # the slices gathered from, in IR order
    planes = {n.inputs[0] for n in slices.values()} - used - set(found)
    return _Tables(ordered, {t for t in ordered if t not in used}, versions, slices, planes)  # type: ignore[arg-type]


# -- names --------------------------------------------------------------------------------------


class _Names:
    """Hands out identifiers that collide with nothing already used (suffix ``_1``, ``_2``, ...)."""

    def __init__(self, taken: "Iterable[str]") -> None:
        self.taken = set(taken)

    def fresh(self, base: str) -> str:
        name, i = base, 0
        while name in self.taken or keyword.iskeyword(name):
            i += 1
            name = f"{base}_{i}"
        self.taken.add(name)
        return name


# -- the two templates ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Extra:
    """A parameter the template adds after the kernel's own and the outputs."""

    name: str
    constexpr: bool
    param: Param  # index is filled in when the signature is assembled


class _Layout:
    """What differs between the templates; the emitter below is shared."""

    template = ""
    grid: "tuple[DimExpr, ...]"

    def extras(self, names: _Names, taken_dims: "set[str]") -> "list[_Extra]":
        raise NotImplementedError

    def prologue(self) -> "list[str]":
        raise NotImplementedError

    def default_num_warps(self) -> int:
        raise NotImplementedError

    def bucket_num_warps(self) -> "dict[str, int]":
        """The warp count the template's rule gives each bucket of a bucketed dimension, by stem suffix (``".D96"``)."""
        return {}

    def header(self) -> "list[str]":
        raise NotImplementedError

    def load(self, p: str, v: Value) -> str:
        raise NotImplementedError

    def store(self, p: str, v: Value, value: str) -> str:
        raise NotImplementedError

    def reduce(self, op: str, a: str) -> str:
        raise NotImplementedError

    def iota(self, v: Value) -> str:
        """The index each lane of ``v`` (the result of ``iota``) holds along the axis it counts, as ``i32``."""
        raise NotImplementedError

    # -- tables (gather_rows, scatter_rows) --------------------------------------------------------------------------------------

    tables: _Tables
    rows_name: "dict[Value, str]"

    def table_extras(self, names: _Names) -> "list[_Extra]":
        """One parameter per table, after the template's own: its number of rows ``N``, which an index is checked against. A ``tl.constexpr``
        when ``N`` is static or bucketed (one binary per bucket), a runtime ``i32`` when it is dynamic."""
        self.rows_name = {}
        self.table_bucket_dims: "list[DimExpr]" = []
        out = []
        by_plane: dict[Any, str] = {}  # the slices of one parameter share its rows parameter (note 19)
        for t in self.tables.params:
            n = t.type.shape[0].folded()
            plane = self.tables.slices[t].inputs[0] if t in self.tables.slices else None
            if plane is not None and plane in by_plane:
                self.rows_name[t] = by_plane[plane]
                continue
            nm = self.rows_name[t] = names.fresh(f"{plane.name if plane is not None else t.name}_rows")  # type: ignore[union-attr]
            if plane is not None:
                by_plane[plane] = nm
            if n.is_static:
                out.append(_Extra(nm, True, _const_param(nm, n.value)))
            elif all(d.buckets for d in n.free_dims()):
                out.append(_Extra(nm, True, _const_param(nm, n)))
                self.table_bucket_dims.append(n)  # (a binary per bucket of it too: the stems of the specs name it)
            else:
                out.append(_Extra(nm, False, Param(nm, 0, "dim", dim=n)))
        return out

    def row_offset(self, t: Value, i: str) -> str:
        """The offset of the element (row ``i`` of the table ``t``) that a lane reads or writes: ``i`` for a rank-1 table."""
        return i

    def row_mask(self, v: Value, ok: "str | None") -> "str | None":
        """The mask of a gather that makes ``v`` (or of a scatter of ``v``), with the index check ``ok`` (``None``: none)."""
        raise NotImplementedError


def _const_param(name: str, default: Any) -> Param:
    return Param(name, 0, "const", default=default, keyword_only=True)


@dataclass
class _Phases:
    """Where each node of a chunked row goes: ``"pre"`` (before the loop over the chunks: per-row values that need no reduction), ``"loop"``
    (per element, once per chunk; an argmax / argmin also folds each chunk into its carry there) or ``"post"`` (after the loop: what depends
    on the result of an argmax / argmin)."""

    values: "dict[Value, str]"
    nodes: "dict[Node, str]"


_PHASE_ORDER = {"pre": 0, "loop": 1, "post": 2}


def _chunk_phases(graph: Graph, cls: "dict[Value, str]", tables: _Tables) -> "_Phases | None":
    """The phases of a chunked row, or ``None`` when the body cannot be chunked (a per-element value depends on an argmax result)."""
    values: dict[Value, str] = {}
    nodes: dict[Node, str] = {}

    def phase(v: Any) -> str:
        return values.get(v, "pre") if isinstance(v, Value) else "pre"

    for p in graph.inputs:
        if not tables.skip(p) and not p.type.devscalar:
            values[p] = "loop" if cls[p] in (_FULL, _COL) else "pre"
    for n in graph.nodes:
        ins = [i for i in n.inputs if isinstance(i, Value) and not tables.skip(i)]
        if n.op == "store":
            ins = [n.inputs[1]]  # type: ignore[list-item]
        elif n.op == "scatter_rows":
            ins = list(n.inputs[1:])  # type: ignore[arg-type]
        latest = max((phase(i) for i in ins), key=_PHASE_ORDER.__getitem__, default="pre")
        per_element = any(cls.get(o) in (_FULL, _COL) for o in n.outputs) or (n.op == "scatter_rows" and cls[n.inputs[2]] in (_FULL, _COL))  # type: ignore[index]
        if n.op == "store":
            per_element = cls[n.inputs[1]] in (_FULL, _COL)  # type: ignore[index]
        if n.op in _ARG_REDUCTIONS:
            nodes[n] = "loop"
            values[n.outputs[0]] = "post"
            continue
        if per_element:
            if latest == "post":
                return None
            here = "loop"
        else:
            if latest == "loop":  # a per-row value from per-element ones without a reduction: not a case the template has
                return None
            here = latest
        nodes[n] = here
        for o in n.outputs:
            values[o] = here
    for v in graph.outputs:
        if cls[v] in (_FULL, _COL) and values.get(v, "pre") == "post":  # pragma: no cover - a per-element post value returned early
            return None
    return _Phases(values, nodes)


class _RowLayout(_Layout):
    template = "row"

    def __init__(self, op: KernelOp, tables: _Tables, lowering: int = 1) -> None:
        graph, name = op.graph, op.name
        self.tables = tables
        self.lowering = lowering
        names = _strided_params(op)
        if names and lowering < 2:
            raise _fail(name, "row", "row.strided", "a fused region that reads a view where it is is lowered by the elementwise template")
        values = [v for v in _all_values(graph) if not tables.skip(v)]
        for v in values:
            if v.type.rank > 2:
                raise _fail(
                    name,
                    "row",
                    "row.rank",
                    f"{_label(v)} has rank {v.type.rank}; the row template takes [R, D], [D], [R] and [R, 1] (rank 1 or 2)",
                )
        two = [v for v in values if v.type.rank == 2]
        if not two:
            raise _fail(name, "row", "row.no-rows", "the row template needs a rank-2 tensor [R, D]; the kernel has none")
        pick = next((v for v in two if not _is_one(v.type.shape[1])), two[0])
        self.R, self.D = pick.type.shape[0], pick.type.shape[1].folded()
        D = self.D
        if not D.is_static and not all(d.buckets for d in D.free_dims()):
            dynamic = ", ".join(sorted(d.name for d in D.free_dims() if not d.buckets))
            raise _fail(
                name,
                "row",
                "row.dynamic-d",
                f"the row length D = {D} (last axis of {_label(pick)}) is dynamic in {dynamic}; it is baked into the "
                "binary, so it must be static (Dim('D', 2560)) or bucketed (Dim('D', buckets=[...]))",
            )
        for n in graph.nodes:
            if n.op in _REDUCTIONS or n.op in _ARG_REDUCTIONS:
                src = n.inputs[0]
                assert isinstance(src, Value)
                axes = n.attrs["axes"]
                if axes != (src.type.rank - 1,):
                    raise _fail(
                        name,
                        "row",
                        "row.last-axis-reduction",
                        f"{n.op} over axes {axes} of {_label(src)}: reductions run over the last axis only",
                    )
                if src.type.rank != 2:
                    raise _fail(
                        name,
                        "row",
                        "row.reduction-input",
                        f"{n.op} of {_label(src)}: a reduction needs an [R, D] input, this tensor has no row axis "
                        "to reduce along",
                    )
        for v in values:
            if v.type.rank == 0 and v.is_param and not v.type.devscalar:  # a loaded DevScalar is a rank-0 value, and so is what is computed from it
                raise _fail(
                    name,
                    "row",
                    "row.rank",
                    f"{_label(v)} has rank 0; the row template takes [R, D], [D], [R] and [R, 1] (rank 1 or 2), and a DevScalar",
                )
        self.block_max = _max_block(name, D)
        self.cls = {}
        for v in values:
            shape = v.type.shape
            if len(shape) == 0:
                self.cls[v] = _DEV if v.type.devscalar else _SCALAR
            elif len(shape) == 2:
                if shape[0] == self.R and shape[1] == D:
                    self.cls[v] = _FULL
                elif shape[0] == self.R and _is_one(shape[1]):
                    self.cls[v] = _ROW
                else:
                    raise self._shape_error(name, v)
            else:
                if shape[0] == D and shape[0] == self.R:
                    raise _fail(
                        name,
                        "row",
                        "row.ambiguous",
                        f"{_label(v)}: R and D are both {D}, so a rank-1 tensor could be [D] or [R]; give them different sizes",
                    )
                if shape[0] == D:
                    self.cls[v] = _COL
                elif shape[0] == self.R:
                    self.cls[v] = _ROW
                else:
                    raise self._shape_error(name, v)
        for n in graph.nodes:
            if n.op in _REDUCTIONS or n.op in _ARG_REDUCTIONS:
                src = n.inputs[0]
                assert isinstance(src, Value)
                if self.cls[src] != _FULL:
                    raise _fail(
                        name,
                        "row",
                        "row.reduction-input",
                        f"{n.op} of {_label(src)}: a reduction needs an [R, D] input, this tensor is one value per row",
                    )
        for p in graph.inputs:
            if p.access != "in" and self.cls.get(p) == _COL:
                raise _fail(
                    name,
                    "row",
                    "row.col-output",
                    f"{p.access} parameter '{p.name}' ({p.type}) has shape [D]; every row program would write it "
                    "(a race), so only [R, D] and per-row tensors can be written",
                )
        for v in graph.outputs:
            if self.cls[v] == _COL:
                raise _fail(
                    name,
                    "row",
                    "row.col-output",
                    f"the returned {_label(v)} has shape [D]; every row program would write it (a race), so only "
                    "[R, D] and per-row tensors can be returned",
                )
            if self.cls[v] in (_SCALAR, _DEV):
                raise _fail(
                    name,
                    "row",
                    "row.scalar-output",
                    f"the returned {_label(v)} is a scalar (read from a DevScalar); combine it with a tensor first",
                )
        for n in graph.nodes:
            if n.op == "scatter_rows" and self.cls[n.inputs[1]] == _COL:  # type: ignore[index]
                raise _fail(
                    name,
                    "row",
                    "row.scatter-index",
                    f"{n}: the index has shape [D], so every row program would write the same elements (a race); a scatter's index "
                    "is [R] (one row each) or [R, D]",
                )
        self.grid = (self.R.folded(),)
        # revision 2, context-free (a fused region, note 21): a view read where it is, row r column c at r * s0 + c * s1 ([R, D]), c * s0 ([D]),
        # r * s0 ([R], [R, 1]); a view is read, never written, and is not a table
        self.strided: dict[Value, list[str]] = {}
        for v in graph.inputs:
            if v.name not in names:
                continue
            if v.access != "in" or v in tables.params or tables.skip(v) or v in tables.slices or self.cls.get(v) not in (_FULL, _COL, _ROW):
                # (a table's rows are addressed by an index, contiguously: a view gathered from is copied first)
                raise _fail(name, "row", "row.strided", f"the view {_label(v)} is read where it is only as an [R, D], [D], [R] or [R, 1] operand, never gathered from")
            self.strided[v] = names[v.name]
        # revision 2 (note 21): a row whose reductions are ordered sums (and order-free ones) takes the warps of its BLOCK_D by the rule of
        # revision 2, a speed choice; a Triton tree reduction (sum or mean without order) keeps the template's rule, which is its numerics (note 6)
        # (a sequential order, one lane, is revision 1's helper whatever the revision: the template's rule)
        self.ordered_rule = (
            lowering >= 2
            and any(n.op in ("sum", "mean") and "order" in n.attrs for n in graph.nodes)
            and not any(n.op in ("sum", "mean") and ("order" not in n.attrs or n.attrs["order"].lanes == 1) for n in graph.nodes)
        )
        self.order_lanes = max((n.attrs["order"].lanes for n in graph.nodes if n.op in ("sum", "mean") and "order" in n.attrs), default=1)
        # an argmax or argmin of a long row walks it in chunks (note 8): each lane keeps its best (value, index) over the chunks, then one
        # tree reduce. Only where the order cannot matter: no sum, mean or max in the kernel, and nothing per element depends on a result
        self.chunk: "int | None" = None
        self.phases: "_Phases | None" = None
        if any(n.op in _ARG_REDUCTIONS for n in graph.nodes) and not any(n.op in _REDUCTIONS for n in graph.nodes):
            lo, hi = dim_bounds(D)
            if lo > ARG_CHUNK and hi is not None and hi <= ARG_CHUNK * ARG_MAX_CHUNKS:
                self.phases = _chunk_phases(graph, self.cls, tables)
                if self.phases is not None:
                    self.chunk = ARG_CHUNK
                    self.block_max = ARG_CHUNK

    def _shape_error(self, name: str, v: Value) -> TilefishLoweringError:
        return _fail(
            name,
            "row",
            "row.shape",
            f"{_label(v)} does not fit: with R = {self.R} and D = {self.D} the row template takes [{self.R}, {self.D}], "
            f"[{self.D}], [{self.R}] and [{self.R}, 1]",
        )

    # -- layout protocol ------------------------------------------------------------------------

    def extras(self, names: _Names, taken_dims: "set[str]") -> "list[_Extra]":
        self.D_name = names.fresh("D")
        self.BLOCK_name = names.fresh("BLOCK_D")
        self.row, self.cols, self.mask, self.offs = (names.fresh(x) for x in ("row", "cols", "mask", "offs"))
        if self.chunk is not None:
            self.lanes, self.c0 = names.fresh("lanes"), names.fresh("c0")
        if self.D.is_static:
            d_default: Any = self.D.value
            block_default: Any = next_pow2(self.D.value)
        else:
            d_default = self.D
            block_default = next_pow2(self.D)
        if self.chunk is not None:
            block_default = self.chunk
        return [
            _Extra(self.D_name, True, _const_param(self.D_name, d_default)),
            _Extra(self.BLOCK_name, True, _const_param(self.BLOCK_name, block_default)),
            *self.table_extras(names),
            *self.slice_extras(names),
        ]

    def slice_extras(self, names: _Names) -> "list[_Extra]":
        """One runtime ``i32`` per dynamic dimension that the offset of a static slice needs (note 19): ``size_R`` for ``codes[3]`` of a ``[K, R]``
        parameter, which a host derives from the shapes like any dimension scalar. A static offset is a number."""
        self.size_names: dict[Dim, str] = {}
        out = []
        for v in self.tables.slices:
            for atom, _ in sorted(v.type.offset.folded().terms, key=lambda t: str(t[0])):  # type: ignore[union-attr]
                assert isinstance(atom, Dim)  # _static_slice_problem
                if atom not in self.size_names:
                    nm = self.size_names[atom] = names.fresh(f"size_{atom.name}")
                    out.append(_Extra(nm, False, Param(nm, 0, "dim", dim=atom)))
        return out

    def slice_offset(self, v: Value) -> str:
        """The element offset of the static slice ``v`` in its parameter, as Triton source: ``3 * size_R``, a number, or ``""`` for 0."""
        offset = v.type.offset.folded()  # type: ignore[union-attr]
        terms = [f"{c} * {self.size_names[a]}" if c != 1 else self.size_names[a] for a, c in sorted(offset.terms, key=lambda t: str(t[0]))]  # type: ignore[index]
        if offset.offset:
            terms.append(str(offset.offset))
        return " + ".join(terms)

    def row_offset(self, t: Value, i: str) -> str:
        return f"{i} * {self.D_name} + {self.cols}" if t.type.rank == 2 else i

    def row_mask(self, v: Value, ok: "str | None") -> "str | None":
        if self.cls[v] == _ROW:  # one element per row program: only the index check
            return ok
        return self.mask if ok is None else f"{self.mask} & {ok}"

    def prologue(self) -> "list[str]":
        return [
            f"{self.row} = tl.program_id(0)",
            f"{self.cols} = tl.arange(0, {self.BLOCK_name})",
            f"{self.mask} = {self.cols} < {self.D_name}",
            f"{self.offs} = {self.row} * {self.D_name} + {self.cols}",
        ]

    def chunk_prologue(self) -> "list[str]":
        """A chunked row: what comes before the loop over the chunks."""
        return [f"{self.row} = tl.program_id(0)", f"{self.lanes} = tl.arange(0, {self.BLOCK_name})"]

    def chunk_loop(self) -> "list[str]":
        """A chunked row: the loop over the chunks (unrolled; the columns in ascending order) and the first lines of its body."""
        return [
            f"for {self.c0} in tl.static_range(0, {self.D_name}, {self.BLOCK_name}):",
            f"    {self.cols} = {self.c0} + {self.lanes}",
            f"    {self.mask} = {self.cols} < {self.D_name}",
            f"    {self.offs} = {self.row} * {self.D_name} + {self.cols}",
        ]

    def warps_of(self, block: int) -> int:
        """The warp count of a row of ``BLOCK_D = block``: the template's rule (note 6), or revision 2's for a row of ordered sums (note 21)."""
        return _ordered_row_warps(block, self.order_lanes) if self.ordered_rule else _row_warps(block)

    def default_num_warps(self) -> int:
        if self.chunk is not None:
            return ARG_CHUNK_WARPS
        return self.warps_of(self.block_max)

    def bucket_num_warps(self) -> "dict[str, int]":
        """One warp count per bucket of ``D``, from the ``BLOCK_D`` of that bucket alone (none for a static ``D``)."""
        if self.D.is_static or self.chunk is not None:  # a chunked row has the same block in every bucket
            return {}
        d_dims = sorted(self.D.free_dims(), key=lambda x: x.name)
        # the order compile_specs walks the buckets in: the constexprs' dims in order of appearance, D's first, then a bucketed table's rows
        free = list(d_dims)
        for n in getattr(self, "table_bucket_dims", []):
            free += [x for x in sorted(n.free_dims(), key=lambda x: x.name) if x not in free]
        out: dict[str, int] = {}
        for combo in itertools.product(*(x.buckets for x in free)):  # type: ignore[misc]
            value = self.D.subst(dict(zip(free[: len(d_dims)], combo[: len(d_dims)]))).value
            out["".join(f".{x.name}{v}" for x, v in zip(free, combo))] = self.warps_of(next_pow2(value))
        return out

    def header(self) -> "list[str]":
        d = f"{self.D} (static)" if self.D.is_static else f"{self.D} (bucketed)"
        slices = [
            f"'{n.inputs[0].name}[{n.attrs['index']}]' {_shape_text(v.type.shape)} is read at {n.inputs[0].name} + {self.slice_offset(v) or 0} (a static slice, note 19)"  # type: ignore[union-attr]
            for v, n in self.tables.slices.items()
        ]
        views = [f"'{v.name}' is a view read where it is: row r, column c at {self.strided_offset(v, 'r', 'c')} (note 21)" for v in self.strided]
        if self.chunk is not None:
            return [
                f"template: row; one program per row, grid {_grid_text(self.grid)}",
                f"D = {d}, walked in chunks of BLOCK_D = {self.chunk} columns (argmax / argmin; note 8)",
                *slices,
                *views,
            ]
        return [f"template: row; one program per row, grid {_grid_text(self.grid)}", f"D = {d}, BLOCK_D = next_pow2(D)", *slices, *views]

    def strided_offset(self, v: Value, row: str, cols: str) -> str:
        """The element of the strided operand ``v`` (revision 2, note 21) at row ``row`` and columns ``cols``: each coordinate it has times the
        stride the call passes (an ``[R, 1]`` view has no column coordinate)."""
        s = self.strided[v]
        c = self.cls[v]
        if c == _FULL:
            return f"{row} * {s[0]} + {cols} * {s[1]}"
        if c == _COL:
            return f"{cols} * {s[0]}"
        return f"{row} * {s[0]}"

    def load(self, p: str, v: Value) -> str:
        c = self.cls[v]
        if v in self.strided:
            off = self.strided_offset(v, self.row, self.cols)
            if c in (_FULL, _COL):
                return f"tl.load({p} + ({off}), mask={self.mask}, other=0)"
            return f"tl.load({p} + {off})"
        if c == _FULL:
            return f"tl.load({p} + {self.offs}, mask={self.mask}, other=0)"
        if c == _COL:
            return f"tl.load({p} + {self.cols}, mask={self.mask}, other=0)"
        return f"tl.load({p} + {self.row})"

    def store(self, p: str, v: Value, value: str) -> str:
        if self.cls[v] == _FULL:
            return f"tl.store({p} + {self.offs}, {value}, mask={self.mask})"
        return f"tl.store({p} + {self.row}, {value})"

    def reduce(self, op: str, a: str) -> str:
        if op == "max":
            return f"tl.max(tl.where({self.mask}, {a}, float(\"-inf\")), axis=0)"
        total = f"tl.sum(tl.where({self.mask}, {a}, 0.0), axis=0)"
        return total if op == "sum" else f"{total} / {self.D_name}"

    def iota(self, v: Value) -> str:
        return self.row if self.cls[v] == _ROW else self.cols  # [R, 1] or [R]: the row; [D]: the column


def _row_warps(block: int) -> int:
    """The row template's warp rule: one warp per 256 lanes of ``BLOCK_D``, between 1 and 16."""
    return max(1, min(ROW_MAX_WARPS, block // ROW_LANES_PER_WARP))


def _ordered_row_warps(block: int, lanes: int) -> int:
    """Revision 2's warp rule of a row whose reductions are ordered (note 21; measured on sm_103), ``lanes`` the most lanes of its orders: one
    warp up to ``BLOCK_D = 2048``, then one warp per 1024 columns (per 512 for an order of 512 lanes or more), at most 16."""
    if block <= ORDERED_ONE_WARP:
        return 1
    per = ORDERED_LANES_PER_WARP // 2 if lanes >= ORDERED_WIDE_LANES else ORDERED_LANES_PER_WARP
    return max(1, min(ROW_MAX_WARPS, block // per))


def _max_block(kernel: str, d: DimExpr) -> int:
    """The largest ``next_pow2(D)`` over the buckets of ``D`` (its value when static)."""
    free = sorted(d.free_dims(), key=lambda x: x.name)
    best = 0
    for combo in itertools.product(*(x.buckets for x in free)):  # type: ignore[misc]
        value = d.subst(dict(zip(free, combo))).value
        if value < 1:
            raise _fail(kernel, "row", "row.dynamic-d", f"D = {d} is {value} in the bucket {dict(zip((x.name for x in free), combo))}")
        best = max(best, next_pow2(value))
    if best > MAX_BLOCK:
        raise _fail(
            kernel, "row", "row.block-size", f"a row of {best} lanes is more than Triton's {MAX_BLOCK}-element tensor limit"
        )
    return best


class _ElementwiseLayout(_Layout):
    template = "elementwise"

    def __init__(self, op: KernelOp, tables: _Tables, broadcast: bool = True) -> None:
        graph, name = op.graph, op.name
        self.tables = tables
        if tables.slices:
            v, n = next(iter(tables.slices.items()))
            raise _fail(
                name,
                "elementwise",
                "elementwise.slice",
                f"{n} is a static slice of the parameter '{n.inputs[0].name}' ({_label(v)}): the row template reads one (note 19), the "  # type: ignore[union-attr]
                "elementwise template does not",
            )
        for t in tables.params:
            if t.type.rank != 1:
                raise _fail(
                    name,
                    "elementwise",
                    "elementwise.table-rank",
                    f"{_label(t)} is a table of rows [N, D]; the elementwise template gathers and scatters elements of an [N] table, a "
                    "row of [N, D] is one program of the row template",
                )
        all_values = [v for v in _all_values(graph) if not tables.skip(v)]
        for v in all_values:
            if v.type.rank == 0 and v.is_param and not v.type.devscalar:
                raise _fail(name, "elementwise", "elementwise.same-shape", f"{_label(v)} has rank 0; give it a shape, or declare a DevScalar")
        values = [v for v in all_values if v.type.rank > 0]  # a DevScalar, what is loaded from it and what is computed from that broadcast
        if not values:
            raise _fail(name, "elementwise", "elementwise.no-tensor", "the kernel has no tensors")
        for n in graph.nodes:
            if n.op in _REDUCTIONS or n.op in _ARG_REDUCTIONS:
                raise _fail(
                    name, "elementwise", "elementwise.reduction", f"{n.op} is a reduction; the elementwise template has none"
                )
        for v in graph.outputs:
            if v.type.rank == 0:
                raise _fail(name, "elementwise", "elementwise.scalar-output", f"the returned {_label(v)} is a scalar; combine it with a tensor first")
        widest = max(values, key=lambda v: v.type.rank)  # the first of the largest rank
        if broadcast:  # the kernel's shape is the one every operand broadcasts to (an operand of the largest rank may itself be broadcast)
            widest = next(
                (w for w in values if w.type.rank == widest.type.rank and all(_kept_axes(v.type.shape, w.type.shape) is not None for v in values)),
                widest,
            )
        self.shape = widest.type.shape
        # a strided parameter (a fused region's view input, note 20) is read at sum(coordinate_k * stride_k): it has the kernel's shape
        self.strided: dict[Value, list[str]] = {}
        names = _strided_params(op)
        for v in graph.inputs:
            if v.name in names:
                if v.type.shape != self.shape or v.access != "in" or v in tables.params:  # (a table is addressed by an index, contiguously)
                    raise _fail(
                        name,
                        "elementwise",
                        "elementwise.strided-shape",
                        f"the view {_label(v)} is read where it is only with the kernel's shape {_shape_text(self.shape)}, and never gathered from",
                    )
                self.strided[v] = names[v.name]
        # an operand whose shape is the trailing axes of the kernel's shape (after leading 1s) broadcasts over the leading ones: lane o reads its
        # element o % n. Any other shape that broadcasts to the kernel's (numpy's rules: a size-1 axis or a missing leading axis repeats) is read
        # at the offset of the lane's coordinates along the axes it has (note 12): ``self.general``
        self.broadcast: dict[Value, str] = {}
        self.general: dict[Value, list[bool]] = {}
        for v in values:
            shape = v.type.shape
            if shape == self.shape:
                continue
            kept = _kept_axes(shape, self.shape) if broadcast else None
            if kept is not None:
                self.broadcast[v] = ""
                if not _is_suffix(kept, self.shape):
                    self.general[v] = kept
                continue
            if not broadcast:
                raise _fail(
                    name,
                    "elementwise",
                    "elementwise.same-shape",
                    f"{_label(v)} does not have the shape {_shape_text(self.shape)} of {_label(widest)}: the "
                    "elementwise template does not broadcast (only python and const scalars combine with a tensor)",
                )
            raise _fail(
                name,
                "elementwise",
                "elementwise.same-shape",
                f"{_label(v)} does not broadcast to the shape {_shape_text(self.shape)} of {_label(widest)}: the elementwise template "
                "broadcasts an operand whose every axis is 1 or the kernel's (numpy's rules, the shapes aligned on their last axes: a [T, D] "
                "operand of a [K, T, D] kernel, a [K, 1, D] one), and python and const scalars",
            )
        for v in [*(p for p in graph.inputs if p.access != "in"), *graph.outputs]:
            if v in self.broadcast:
                raise _fail(
                    name,
                    "elementwise",
                    "elementwise.broadcast-write",
                    f"{_label(v)} is written or returned with the shape {_shape_text(v.type.shape)}, which broadcasts over the kernel's "
                    f"{_shape_text(self.shape)}: every lane that maps onto one of its elements would write it. What is written has the kernel's shape",
                )
        for n in graph.nodes:
            if n.op in ("gather_rows", "scatter_rows") and any(o in self.broadcast for o in (*n.outputs, *n.inputs) if isinstance(o, Value)):
                raise _fail(
                    name,
                    "elementwise",
                    "elementwise.broadcast-index",
                    f"{n}: a gather or a scatter of a broadcast shape is not lowered here; give it the kernel's shape {_shape_text(self.shape)}",
                )
        # the kernel axes whose sizes a coordinate needs (see coordinate): those after a run of axes, and the run's own unless it starts at 0
        r = len(self.shape)
        runs = [run for kept in self.general.values() for run in _runs(kept)]
        runs += [(r - n.outputs[0].type.rank,) * 2 for n in graph.nodes if n.op == "iota" and n.outputs[0] in self.broadcast]
        runs += [(k, k) for _ in self.strided for k in range(r)]  # a strided operand needs the lane's coordinate along every axis
        size_axes: set[int] = set()
        for a, b in runs:
            size_axes.update(range(b + 1, r))
            if a > 0:
                size_axes.update(range(a, b + 1))
        self.size_axes = sorted(k for k in size_axes if not self.shape[k].folded().is_static)
        self.total = _product(self.shape)

    def extras(self, names: _Names, taken_dims: "set[str]") -> "list[_Extra]":
        self.n_name = names.fresh("n")
        self.BLOCK_name = names.fresh("BLOCK")
        self.pid, self.offs, self.mask = (names.fresh(x) for x in ("pid", "offs", "mask"))
        if self.total is not None:
            # the count is the product of the shape, so a spec says how to compute it and derives its divisibility
            self.n_dim = self.total
        else:
            # two symbolic factors: not an affine expression, so a fresh dimension that carries what the factors prove
            dim_name, i = self.n_name, 0
            while dim_name in taken_dims:
                i += 1
                dim_name = f"{self.n_name}_{i}"
            k = min(ELEMENTWISE_BLOCK, math.prod(_pow2_divisor(d) for d in self.shape))
            self.n_dim = Dim(dim_name, divisible_by=k if k > 1 else None)
        self.grid = (as_dim(cdiv(self.n_dim, ELEMENTWISE_BLOCK)),)
        # a broadcast parameter's element count: a runtime i32 that a host reads off the tensor (count_of)
        self.counts: dict[str, str] = {}
        bcast: list[_Extra] = []
        for v in self.broadcast:
            if not v.is_param or v in self.general:  # a general broadcast operand is read at its coordinates' offset: no count
                continue
            nm = self.broadcast[v] = names.fresh(f"{v.name}_n")
            total = _product(v.type.shape)
            dim = total if total is not None else Dim(nm)
            bcast.append(_Extra(nm, False, Param(nm, 0, "dim", dim=dim)))
            self.counts[nm] = v.name
        # the sizes of the kernel axes a coordinate needs: one runtime i32 per distinct symbolic size (a static one is written as a number)
        self.size_names: dict[int, str] = {}
        sizes: list[_Extra] = []
        by_expr: dict[DimExpr, str] = {}
        for k in self.size_axes:
            d = self.shape[k].folded()
            if d not in by_expr:
                nm = by_expr[d] = names.fresh(f"size_{d.name}" if isinstance(d, Dim) else f"size{k}")
                sizes.append(_Extra(nm, False, Param(nm, 0, "dim", dim=d)))
            self.size_names[k] = by_expr[d]
        return [
            _Extra(self.n_name, False, Param(self.n_name, 0, "dim", dim=self.n_dim)),
            _Extra(self.BLOCK_name, True, _const_param(self.BLOCK_name, ELEMENTWISE_BLOCK)),
            *bcast,
            *sizes,
            *self.table_extras(names),
        ]

    def _size(self, axes: "Iterable[int]") -> str:
        """The product of the kernel's sizes along ``axes`` as Triton source: the static ones folded into one number, the symbolic ones the
        names of their runtime ``i32`` parameters (``"4 * size_L"``; ``"1"`` for none)."""
        static, symbolic = 1, []
        for k in axes:
            d = self.shape[k].folded()
            if d.is_static:
                static *= d.value
            else:
                symbolic.append(self.size_names[k])
        return " * ".join(([str(static)] if static != 1 or not symbolic else []) + symbolic)

    def coordinate(self, lane: str, first: int, last: int) -> str:
        """The index of the lane ``lane`` along the kernel axes ``first`` to ``last`` taken as one axis (row-major): ``(lane // inner) % size``,
        without the division for the last axes and without the remainder for the first ones (a lane is below the element count)."""
        inner = self._size(range(last + 1, len(self.shape)))
        out = lane if inner == "1" else f"{lane} // ({inner})" if " * " in inner else f"{lane} // {inner}"
        if first > 0:
            size = self._size(range(first, last + 1))
            out = (f"({out})" if out != lane else out) + (f" % ({size})" if " * " in size else f" % {size}")
        return out

    def broadcast_offset(self, v: Value, lane: str) -> str:
        """The element of the general broadcast operand ``v`` that the lane ``lane`` reads (note 12): the sum, over each run of the axes ``v`` has,
        of the lane's coordinate along the run times ``v``'s own stride there (the product of its sizes after the run)."""
        kept = self.general[v]
        terms = []
        for a, b in _runs(kept):
            coord = self.coordinate(lane, a, b)
            stride = self._size(k for k in range(b + 1, len(self.shape)) if kept[k])
            terms.append(coord if stride == "1" else f"({coord}) * {stride}")
        return " + ".join(terms) if terms else "0"

    def row_mask(self, v: Value, ok: "str | None") -> "str | None":
        return self.mask if ok is None else f"{self.mask} & {ok}"

    def prologue(self) -> "list[str]":
        return [
            f"{self.pid} = tl.program_id(0)",
            f"{self.offs} = {self.pid} * {self.BLOCK_name} + tl.arange(0, {self.BLOCK_name})",
            f"{self.mask} = {self.offs} < {self.n_name}",
        ]

    def default_num_warps(self) -> int:
        return ELEMENTWISE_NUM_WARPS

    def header(self) -> "list[str]":
        n = str(self.total) if self.total is not None else " * ".join(str(d) for d in self.shape)
        lines = [
            f"template: elementwise; every tensor is {_shape_text(self.shape)}, flattened to n = {n} elements",
            f"grid {_grid_text(self.grid)}, BLOCK = {ELEMENTWISE_BLOCK}",
        ]
        for v in self.strided:
            lines.append(f"'{v.name}' is a view read where it is: lane o reads its element {self.strided_offset(v, 'o')} (note 20)")
        for v, count in self.broadcast.items():
            if v.is_param and v in self.general:
                lines.append(
                    f"'{v.name}' {_shape_text(v.type.shape)} broadcasts over the axes where it has size 1: lane o reads its element "
                    f"{self.broadcast_offset(v, 'o')}"
                )
            elif v.is_param:
                lines.append(f"'{v.name}' {_shape_text(v.type.shape)} broadcasts over the leading axes: lane o reads its element o % {count}")
        return lines

    def strided_offset(self, v: Value, lane: str) -> str:
        """The element of the strided operand ``v`` that the lane ``lane`` reads: its coordinate along each axis times the stride the call passes."""
        terms = []
        for k, stride in enumerate(self.strided[v]):
            coord = self.coordinate(lane, k, k)
            terms.append(f"({coord}) * {stride}")  # (a stride parameter is named a<k>_s<axis>, which no reserved name is)
        return " + ".join(terms)

    def load(self, p: str, v: Value) -> str:
        if v in self.strided:
            return f"tl.load({p} + ({self.strided_offset(v, self.offs)}), mask={self.mask}, other=0)"
        if v in self.general:
            return f"tl.load({p} + ({self.broadcast_offset(v, self.offs)}), mask={self.mask}, other=0)"
        if v in self.broadcast:
            return f"tl.load({p} + {self.offs} % {self.broadcast[v]}, mask={self.mask}, other=0)"
        return f"tl.load({p} + {self.offs}, mask={self.mask}, other=0)"

    def store(self, p: str, v: Value, value: str) -> str:
        return f"tl.store({p} + {self.offs}, {value}, mask={self.mask})"

    def reduce(self, op: str, a: str) -> str:  # pragma: no cover - rejected in __init__
        raise AssertionError("the elementwise template has no reductions")

    def iota(self, v: Value) -> str:
        if v not in self.broadcast:
            return self.offs  # the result has the kernel's shape, which is then rank 1
        # [S, 1, ..., 1] aligned on the kernel's last axes: the lane's coordinate along kernel axis k (note 12)
        k = len(self.shape) - v.type.rank
        if _is_one(v.type.shape[0]) and not _is_one(self.shape[k]):
            return f"{self.offs} * 0"  # an axis of size 1: every lane is at index 0
        return self.coordinate(self.offs, k, k)


def _kept_axes(shape: "tuple[DimExpr, ...]", kernel: "tuple[DimExpr, ...]") -> "list[bool] | None":
    """Whether an operand of ``shape`` broadcasts to ``kernel`` (numpy's rules: aligned on the last axes, an axis of size 1 or a missing leading axis
    repeats) and, if so, per axis of ``kernel``, whether the operand has it (its size is the kernel's) or repeats along it. ``None``: it does not."""
    r, s = len(kernel), len(shape)
    if s > r:
        return None
    out = []
    for k in range(r):
        j = k - (r - s)
        if j < 0:
            out.append(False)
        elif shape[j] == kernel[k]:
            out.append(True)
        elif _is_one(shape[j]):
            out.append(False)
        else:
            return None
    return out


def _is_suffix(kept: "list[bool]", kernel: "tuple[DimExpr, ...]") -> bool:
    """Whether the axes an operand has are the kernel's last ones (an axis of size 1 of the kernel counts either way): the operand is then read
    at the lane's index modulo its own element count."""
    seen = False
    for has, d in zip(kept, kernel):
        if _is_one(d):
            continue
        if has:
            seen = True
        elif seen:
            return False
    return True


def _runs(kept: "list[bool]") -> "list[tuple[int, int]]":
    """The maximal runs ``(first, last)`` of the axes an operand has."""
    out: list[tuple[int, int]] = []
    start = None
    for k, has in enumerate([*kept, False]):
        if has and start is None:
            start = k
        elif not has and start is not None:
            out.append((start, k - 1))
            start = None
    return out


def _product(shape: "tuple[DimExpr, ...]") -> "DimExpr | None":
    """``prod(shape)`` as a dimension expression, or ``None`` when it is not affine (two symbolic factors)."""
    total = DimExpr.const(1)
    try:
        for d in shape:
            total = total * d
    except TilefishTypeError:
        return None
    return total.folded()


def _pow2_divisor(d: DimExpr) -> int:
    """The largest power of two, up to the elementwise block size, that provably divides ``d``."""
    k = 1
    while k * 2 <= ELEMENTWISE_BLOCK and d.is_divisible_by(k * 2):
        k *= 2
    return k


def _grid_text(grid: "tuple[DimExpr, ...]") -> str:
    from .decorators import grid_to_string

    return grid_to_string(grid)


# -- code generation ------------------------------------------------------------------------------


def _scalar_text(s: Scalar, rename: "dict[str, str]") -> str:
    if s.name is not None:
        return rename.get(s.name, s.name)
    v = s.value
    if isinstance(v, float):
        if math.isnan(v):
            return 'float("nan")'
        if math.isinf(v):
            return 'float("inf")' if v > 0 else 'float("-inf")'
    return repr(v)


def _is_negative_zero(s: Scalar) -> bool:
    """Whether ``s`` is a literal ``-0.0`` of a float dtype (f32, bf16, f16)."""
    v = s.value
    return s.name is None and isinstance(v, float) and v == 0.0 and math.copysign(1.0, v) < 0 and s.dtype.name in ("f32", "bf16", "f16")


def _round_float(value: float, dtype: str) -> float:
    """A python float rounded to nearest even in ``dtype`` (``f32``, ``bf16`` or ``f16``) through f32, as torch rounds one; NaN stays NaN."""
    if math.isnan(value) or math.isinf(value):
        return value
    try:
        (f,) = struct.unpack("<f", struct.pack("<f", value))
    except OverflowError:  # rounds beyond the largest f32
        return math.copysign(math.inf, value)
    if dtype == "f32":
        return f
    if dtype == "f16":
        try:
            (h,) = struct.unpack("<e", struct.pack("<e", f))
        except OverflowError:
            return math.copysign(math.inf, f)
        return float(h)
    assert dtype == "bf16", dtype
    (b,) = struct.unpack("<I", struct.pack("<f", f))
    b = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16 & 0xFFFFFFFF  # round to nearest even on the upper 16 bits (overflow gives inf)
    (r,) = struct.unpack("<f", struct.pack("<I", b))
    return r


def _is_self_store(n: Node) -> bool:
    """``x[...] = x``: the stored value is the version the store replaces."""
    return n.inputs[0] is n.inputs[1]


class _Body:
    """Emits the statements of the kernel body, one local per SSA value."""

    def __init__(
        self,
        op: KernelOp,
        layout: _Layout,
        names: _Names,
        out_names: "list[str]",
        rename: "dict[str, str]",
        math: str = "fast",
        context_free: bool = False,
        fp_fusion: bool = True,
        lowering: int = 1,
    ) -> None:
        self.graph = op.graph
        self.context_free = context_free  # a fused region's kernel (note 20)
        self.lowering = lowering  # the revision of the lowering (note 21)
        # separate roundings (note 2): every float + - * one scalar .rn instruction, every narrowing float cast inline PTX
        self.strict = context_free or not fp_fusion
        self.layout = layout
        self.names = names
        self.out_names = out_names
        self.rename = rename  # kernel parameter name -> its name in the generated function, where they differ
        self.math = math  # the kernel's f32 math (note 13); an op may say its own
        self.libdevice = False  # whether the body calls libdevice by name (tanh), which the module then imports
        self.params = {p.name: p for p in op.graph.inputs}
        self.producer = {o: n for n in op.graph.nodes for o in n.outputs}  # the node that made each value (an fma sum reads its product's operands)
        # a product that only fma sums read is not computed on its own: each step of their chains multiplies (note 17)
        users: dict[Value, list[Node]] = {}
        for n in op.graph.nodes:
            for i in n.inputs:
                if isinstance(i, Value):
                    users.setdefault(i, []).append(n)
        self.fused_only = {
            self.producer[v]
            for v, ns in users.items()
            if v in self.producer and all(u.op == "sum" and u.attrs.get("fma") for u in ns) and not any(v is o for o in op.graph.outputs)
        }
        # a float product that an ordered sum without fma adds is rounded first, whatever fp_fusion says (note 17): one .rn instruction
        self.rn_products = {
            self.producer[v]
            for v, ns in users.items()
            if v in self.producer
            and self.producer[v].op == "mul"
            and any(u.op in ("sum", "mean") and "order" in u.attrs and not u.attrs.get("fma") for u in ns)
        }
        self.var: dict[Value, str] = {}
        self.lines: list[str] = []
        self.made_for: "list[Any]" = []  # per line of ``lines``: the node (or a label of code of no op) it was emitted for
        self.current: Any = None
        self.helpers: dict[tuple, tuple[str, list[str]]] = {}  # module-level @triton.jit functions the body calls (an argmax combine)
        self.indent = 1
        self._count = 0

    def local(self) -> str:
        name = self.names.fresh(f"v{self._count}")
        self._count += 1
        return name

    def widened(self, x: "Value | Scalar") -> bool:
        """Whether ``x`` is, or a ``where`` (through any number of them) selects, a 16-bit float widened by a cast: what MLIR and LLVM fold a
        narrowing cast of (``fptrunc(fpext(x))`` is ``x``, and a ``select`` of widened values is narrowed into a select of the values), which
        keeps an input NaN's payload where the cast gives the canonical NaN (note 1; the RFC's hazard H1)."""
        seen: set[int] = set()
        todo = [x]
        while todo:
            v = todo.pop()
            if not isinstance(v, Value) or id(v) in seen:
                continue
            seen.add(id(v))
            n = self.producer.get(v)
            if n is None:
                continue
            src = n.inputs[0] if n.inputs else None
            if n.op == "cast" and v.type.dtype.name == "f32" and isinstance(src, Value) and src.type.dtype.name in ("bf16", "f16"):
                return True
            if n.op == "where":
                todo += [n.inputs[1], n.inputs[2]]
        return False

    def emit(self, text: str, comment: str = "") -> None:
        self.lines.append("    " * self.indent + text + (f"  # {comment}" if comment else ""))
        self.made_for.append(self.current)

    def operand(self, x: "Value | Scalar") -> str:
        if isinstance(x, Value):
            return self.var[x]
        if self.lowering >= 2 and _is_negative_zero(x):  # note 21: Triton 3.5.1 writes a literal -0.0 as +0.0; the bits of -0.0, bitcast
            bits, ints, name = {"f32": (-(2**31), "int32", "float32"), "bf16": (-(2**15), "int16", "bfloat16"), "f16": (-(2**15), "int16", "float16")}[x.dtype.name]
            return f"tl.full((), {bits}, tl.{ints}).to(tl.{name}, bitcast=True)"
        return _scalar_text(x, self.rename)

    def typed(self, x: "Value | Scalar") -> str:
        """An operand of a comparison or a ``where``: a scalar of a dtype that is not Triton's own literal type is a typed constant,
        so that it is rounded to its dtype (Triton would compare a bf16 tensor with an f32 literal, unrounded). A literal is written
        already rounded (through f32, as torch rounds a python float), so the constant is exact whatever Triton makes of it."""
        if not isinstance(x, Scalar) or x.dtype.name in _LITERAL_DTYPES:
            return self.operand(x)
        if self.lowering >= 2 and _is_negative_zero(x):  # already of its dtype (note 21)
            return self.operand(x)
        if x.name is None and x.dtype.is_float:
            x = Scalar(_round_float(float(x.value), x.dtype.name), x.dtype)
        return f"tl.cast({self.operand(x)}, tl.{_TL[x.dtype.name]})"

    def _reads(self) -> "tuple[list[Value], dict[str, Node]]":
        """The parameters the body loads, in declaration order, and the last store into each written parameter."""
        graph, layout = self.graph, self.layout
        reads: set[Value] = set()
        last_store: dict[str, Node] = {}
        for n in graph.nodes:
            if n.op == "store":
                # a store's first input is the version it replaces, so the parameter is named by the node
                if not _is_self_store(n):  # copying a tensor onto itself writes nothing
                    last_store[n.attrs["target"]] = n
                reads.add(n.inputs[1])  # type: ignore[arg-type]
            elif n.op == "scatter_rows":  # like a store, its first input is the version it replaces
                reads.update(i for i in n.inputs[1:] if isinstance(i, Value))
            else:
                reads.update(i for i in n.inputs if isinstance(i, Value))
        reads.update(graph.outputs)
        # an Out parameter is never read, a DevScalar only by tilefish.load, a table by its gathers, a plane by its static slices
        skip = layout.tables.only | layout.tables.planes
        loads = [p for p in graph.inputs if p.access != "out" and p in reads and not p.type.devscalar and p not in skip]
        return loads, last_store

    def pointer(self, t: Value) -> str:
        """The pointer to the first element of ``t``: the parameter, or for a static slice its parameter plus its offset (note 19)."""
        n = self.layout.tables.slices.get(t)
        if n is None:
            return self.rename.get(t.name, t.name)
        x = n.inputs[0]
        assert isinstance(x, Value) and isinstance(self.layout, _RowLayout)
        offset = self.layout.slice_offset(t)
        return f"{self.rename.get(x.name, x.name)} + {offset}" if offset else self.rename.get(x.name, x.name)

    def static_slice(self, n: Node) -> None:
        """A static slice ``x[k]`` (note 19): read like a parameter of its shape at its offset, where something reads it as a value (a gather
        from it reads its rows at the index instead)."""
        (out,) = n.outputs
        tables = self.layout.tables
        read = any(
            i is out
            for n2 in self.graph.nodes
            for i in (n2.inputs[1:] if n2.op in ("gather_rows", "scatter_rows") else (n2.inputs[1],) if n2.op == "store" else n2.inputs)
        )
        if not read:
            return
        assert out not in tables.only
        v = self.local()
        self.var[out] = v
        x = n.inputs[0]
        assert isinstance(x, Value)
        self.emit(f"{v} = {self.layout.load(self.pointer(out), out)}", f"{x.name}[{n.attrs['index']}]")

    def load(self, p: Value) -> None:
        v = self.local()
        self.var[p] = v
        self.current = f"load {p.name}"
        self.emit(f"{v} = {self.layout.load(self.rename.get(p.name, p.name), p)}", p.name)
        self.current = None

    def node(self, n: Node, last_store: "dict[str, Node]") -> None:
        if n in self.fused_only:  # a product only fma sums read: they multiply in their chains
            return
        self.current = n
        try:
            self._node(n, last_store)
        finally:
            self.current = None

    def _node(self, n: Node, last_store: "dict[str, Node]") -> None:
        if n.op == "store":
            self.store(n, last_store)
        elif n.op == "scatter_rows":
            self.scatter(n)
        elif n.op == "select":
            self.static_slice(n)
        else:
            (out,) = n.outputs
            v = self.local()
            self.var[out] = v
            rounding = "rounding point" if n.is_rounding_point else ""
            if n.op in _IEEE and self.math_of(n) == "ieee" and self.math != "ieee":
                rounding = "math: ieee"  # an op that says its own math in a kernel of the default math
            self.emit(f"{v} = {self.expression(n, v)}", rounding)  # an expression may emit the statements it needs first

    def run(self) -> "list[str]":
        graph, layout = self.graph, self.layout
        if isinstance(layout, _RowLayout) and layout.chunk is not None:
            return self.run_chunked()
        for line in layout.prologue():
            self.emit(line)
        loads, last_store = self._reads()
        for p in loads:  # inputs first, in declaration order
            self.load(p)
        for n in graph.nodes:
            self.node(n, last_store)
        for k, v in enumerate(graph.outputs):
            self.current = f"return value {k}"
            self.emit(layout.store(self.out_names[k], v, self.var[v]), f"return value {k}")
        self.current = None
        return self.lines

    def run_chunked(self) -> "list[str]":
        """A row walked in chunks of ``BLOCK_D`` columns (note 8): the per-row values first, then a loop over the chunks in which every
        per-element value is computed, stored and folded into the carry of each argmax / argmin, then the tree reduce of each carry and
        what depends on the results."""
        graph, layout = self.graph, self.layout
        assert isinstance(layout, _RowLayout) and layout.phases is not None
        phases = layout.phases
        for line in layout.chunk_prologue():
            self.emit(line)
        loads, last_store = self._reads()
        for p in loads:
            if phases.values[p] == "pre":
                self.load(p)
        for n in graph.nodes:
            if phases.nodes[n] == "pre":
                self.node(n, last_store)
        args = [n for n in graph.nodes if n.op in _ARG_REDUCTIONS]
        carries: dict[Node, tuple[str, str, str]] = {}
        for n in args:
            v = self.local()
            self.var[n.outputs[0]] = v
            carries[n] = (v, self.names.fresh(f"{v}_bv"), self.names.fresh(f"{v}_bi"))
            value_type, sentinel, _ = self.arg_kind(n)
            self.emit(f"{carries[n][1]} = tl.full([{layout.BLOCK_name}], {sentinel}, tl.{value_type})", "the best value of each lane")
            self.emit(f"{carries[n][2]} = tl.full([{layout.BLOCK_name}], {_INDEX_SENTINEL}, tl.int32)", "and its column")
        loop = layout.chunk_loop()
        self.emit(loop[0], "every chunk, in ascending order")
        self.indent += 1
        for line in loop[1:]:
            self.emit(line.strip())
        for p in loads:
            if phases.values[p] == "loop":
                self.load(p)
        for n in graph.nodes:
            if phases.nodes[n] != "loop":
                continue
            if n in carries:
                v, bv, bi = carries[n]
                self.current = n
                value, index = self.arg_lanes(n, v)
                self.emit(f"{bv}, {bi} = {self.combine(n)}({bv}, {bi}, {value}, {index})")
                self.current = None
            else:
                self.node(n, last_store)
        for k, v in enumerate(graph.outputs):
            if layout.cls[v] in (_FULL, _COL):
                self.current = f"return value {k}"
                self.emit(layout.store(self.out_names[k], v, self.var[v]), f"return value {k}")
                self.current = None
        self.indent -= 1
        for n in args:
            v, bv, bi = carries[n]
            self.current = n
            self.emit(f"{v}_v, {v}_i = tl.reduce(({bv}, {bi}), 0, {self.combine(n)})")
            out = n.outputs[0].type.dtype.name
            self.emit(f"{v} = {v}_i" if out == "i32" else f"{v} = {v}_i.to(tl.{_TL[out]})")
            self.current = None
        for n in graph.nodes:
            if phases.nodes[n] == "post":
                self.node(n, last_store)
        for k, v in enumerate(graph.outputs):
            if layout.cls[v] not in (_FULL, _COL):
                self.current = f"return value {k}"
                self.emit(layout.store(self.out_names[k], v, self.var[v]), f"return value {k}")
                self.current = None
        return self.lines

    def store(self, n: Node, last_store: "dict[str, Node]") -> None:
        _, value = n.inputs
        assert isinstance(value, Value)
        if n.outputs:  # an InOut parameter: reading it from here on reads the stored value
            self.var[n.outputs[0]] = self.var[value]
        name = n.attrs["target"]
        param = self.params[name]
        label = "InOut" if param.access == "inout" else "Out"
        if _is_self_store(n):
            self.emit(f"# {label} {name}: stored onto itself, nothing to write")
        elif last_store[name] is n:
            what = "final version" if param.access == "inout" else "written once"
            self.emit(self.layout.store(self.rename.get(name, name), param, self.var[value]), f"{label} {name}, {what}")
        else:
            self.emit(f"# {label} {name}: superseded by a later store")

    def row_index(self, n: Node, base: str) -> "tuple[str, str | None]":
        """The row a gather or a scatter addresses, as ``i32``, and the name of the check that it is in range (``None`` for ``clamp``,
        whose index always is): the statements that compute them are emitted here (note 9)."""
        t, idx = n.inputs[0], n.inputs[1]
        assert isinstance(t, Value) and isinstance(idx, Value)
        rows = self.layout.rows_name[t]  # the table parameter itself (a scatter replaces the parameter, never a later version)
        a = self.var[idx]
        dt = idx.type.dtype.name
        if dt == "u8":  # zero extension: exact
            wide = self.names.fresh(f"{base}_idx")
            self.emit(f"{wide} = {a}.to(tl.int32)")
            a, dt = wide, "i32"
        if n.attrs["oob"] == "clamp":
            i = self.names.fresh(f"{base}_i")
            self.emit(f"{i} = tl.minimum(tl.maximum({a}, 0), {rows} - 1)" + ("" if dt == "i32" else ".to(tl.int32)"), "clamped into [0, N)")
            return i, None
        ok = self.names.fresh(f"{base}_ok")
        self.emit(f"{ok} = ({a} >= 0) & ({a} < {rows})", "in range")
        if dt == "i32":
            return a, ok
        i = self.names.fresh(f"{base}_i")
        self.emit(f"{i} = {a}.to(tl.int32)", "the offset is i32 where the index is in range (an i64 out of range is masked)")
        return i, ok

    def gather(self, n: Node, v: str) -> str:
        t = n.inputs[0]
        assert isinstance(t, Value)
        i, ok = self.row_index(n, v)
        mask = self.layout.row_mask(n.outputs[0], ok)
        ptr = f"{self.pointer(t)} + {self.layout.row_offset(t, i)}"
        return f"tl.load({ptr})" if mask is None else f"tl.load({ptr}, mask={mask}, other=0)"

    def scatter(self, n: Node) -> None:
        dst, _, src = n.inputs
        assert isinstance(dst, Value) and isinstance(src, Value)
        base = self.names.fresh(f"s_{dst.name}")
        i, ok = self.row_index(n, base)
        mask = self.layout.row_mask(src, ok)
        ptr = f"{self.rename.get(dst.name, dst.name)} + {self.layout.row_offset(dst, i)}"
        store = f"tl.store({ptr}, {self.var[src]})" if mask is None else f"tl.store({ptr}, {self.var[src]}, mask={mask})"
        self.emit(store, f"InOut {dst.name}: rows written at the index")

    def int_division(self, n: Node, v: str) -> str:
        """``x // c`` or ``x % c`` with a positive constant ``c`` and Python's semantics (note 11): Triton's are C's (``sdiv`` truncates,
        ``srem`` has the sign of the dividend), so a negative remainder is moved into ``[0, c)`` and the quotient down by one."""
        x, c = n.inputs
        assert isinstance(x, Value)
        a, k = self.var[x], self.typed(c)
        sym = "//" if n.op == "floordiv" else "%"
        if not x.type.dtype.signed:  # unsigned: C's division is the floor
            return f"{a} {sym} {k}"
        r = self.names.fresh(f"{v}_r")
        self.emit(f"{r} = {a} % {k}", "C's remainder: the sign of the dividend")
        if n.op == "mod":
            return f"tl.where({r} < 0, {r} + {k}, {r})"
        return f"tl.where({r} < 0, {a} // {k} - 1, {a} // {k})"

    def arg_kind(self, n: Node) -> "tuple[str, str, bool]":
        """The Triton type an argmax / argmin compares in (f32 for every float), the value a masked lane holds so that it loses to every
        lane (``-inf`` / ``+inf``, or the integer type's bound: an equal value loses on its index), and whether the values are floats."""
        x = n.inputs[0]
        assert isinstance(x, Value)
        dt = x.type.dtype
        largest = n.op == "argmax"
        if dt.is_float:
            return "float32", 'float("-inf")' if largest else 'float("inf")', True
        lo, hi = dt.int_range()
        return _TL[dt.name], self.typed(Scalar(lo if largest else hi, dt)), False

    def combine(self, n: Node) -> str:
        """The name of the module-level combine of an argmax / argmin (note 8), written into :attr:`helpers` the first time it is needed."""
        _, _, floating = self.arg_kind(n)
        key = (n.op, floating)
        if key not in self.helpers:
            name = self.names.fresh(f"_tf_{n.op}_{'float' if floating else 'int'}")
            better = "gt" if n.op == "argmax" else "lt"
            extreme = "the larger value" if n.op == "argmax" else "the smaller value"
            lines = [
                "@triton.jit",
                f"def {name}(v1, i1, v2, i2):",
                f"    # whether (v1, i1) beats (v2, i2) in {n.op} (note 8 of tilefish.lower): "
                + (f"a NaN beats every number, else {extreme} wins;" if floating else f"{extreme} wins;"),
                "    # equal values" + (" (two NaNs, -0.0 and +0.0)" if floating else "") + " go to the lower index. A strict total order,",
                "    # so the result depends on no reduction tree and no chunking",
            ]
            if floating:
                lines += [
                    "    gt = v1 > v2",
                    "    lt = v1 < v2",
                    "    n1 = v1 != v1",
                    "    n2 = v2 != v2",
                    f"    win = tl.where(n1 != n2, n1, {better} | ((~gt) & (~lt) & (i1 < i2)))",
                ]
            else:
                lines += [f"    win = (v1 {'>' if n.op == 'argmax' else '<'} v2) | ((v1 == v2) & (i1 < i2))"]
            lines += ["    return tl.where(win, v1, v2), tl.where(win, i1, i2)"]
            self.helpers[key] = (name, lines)
        return self.helpers[key][0]

    def math_of(self, n: Node) -> str:
        """The f32 math of an op: what the call said (``tilefish.exp(x, math="ieee")``), else the kernel's (note 13)."""
        return str(n.attrs.get("math", self.math))

    def ieee_helper(self, op: str) -> str:
        """The name of the module-level helper of an op in IEEE math (note 13), written into :attr:`helpers` the first time it is needed."""
        key = ("ieee", op)
        if key not in self.helpers:
            name = self.names.fresh(f"_tf_{op}_ieee")
            params, doc, body = _IEEE[op]
            self.helpers[key] = (name, ["@triton.jit", f"def {name}({params}):", *(f"    # {line}" for line in doc), *(f"    {line}" for line in body)])
        return self.helpers[key][0]

    def ptx_helper(self, key: str) -> str:
        """The name of the module-level helper of an op that is one PTX instruction in either math (note 16), or of an ordered sum's add
        (note 17), written into :attr:`helpers` the first time it is needed."""
        if ("ptx", key) not in self.helpers:
            name = self.names.fresh(f"_tf_{key.replace('.', '_')}")
            params, doc, body = _PTX[key]
            self.helpers[("ptx", key)] = (name, ["@triton.jit", f"def {name}({params}):", *(f"    # {line}" for line in doc), *(f"    {line}" for line in body)])
        return self.helpers[("ptx", key)][0]

    def first_helper(self) -> str:
        """A combine that keeps its first operand: ``tl.reduce`` of a one-lane tensor with it is that lane's value, bit for bit."""
        if ("first",) not in self.helpers:
            name = self.names.fresh("_tf_first")
            self.helpers[("first",)] = (name, ["@triton.jit", f"def {name}(a, b):", "    # the value of a one-lane tensor as it is (no add)", "    return a"])
        return self.helpers[("first",)][0]

    def sum_helper(self, order: Any, fused: bool) -> str:
        """The name of the module-level helper of an ordered sum (note 17): ``order.lanes`` chains over the columns of the row, in the order the
        order says, then its tree over the lanes; ``fused``: each step of a chain is ``fma(a, b, acc)``. Revision 2 of the lowering takes the
        chains by halving splits where a lane has few enough steps (note 21), the same adds in the same order."""
        if self.lowering >= 2 and order.lanes > 1:
            return self.split_sum_helper(order, fused)
        return self.gather_sum_helper(order, fused)

    def chain_helper(self, fused: bool) -> str:
        """Revision 2 (note 21): the helper that adds the columns of a ``[L, K]`` tensor into the ``[L]`` chains ``p`` in the order of the last axis
        (``K`` a power of two): the low half first, then the high half, each by a halving split (``tl.split`` of a reshape and a permute: data
        movement, exact), down to one column, each one rounding (``add.rn``, or ``fma.rn`` of the two operands of a product)."""
        key = ("chain", fused)
        if key in self.helpers:
            return self.helpers[key][0]
        name = self.names.fresh("_tf_chain_fma" if fused else "_tf_chain")
        if fused:
            step = self.ptx_helper("fma")
            lines = [
                "@triton.jit",
                f"def {name}(p, a, b, L: tl.constexpr, K: tl.constexpr):",
                "    # the chains p[j] = fma(a[j, k], b[j, k], p[j]) for k = 0, 1, ..., K - 1 in order (note 21 of tilefish.lower): the columns of the low",
                "    # half, then of the high half, each half taken by a split (data movement); K is a power of two",
                "    if K == 1:",
                f"        p = {step}(tl.reshape(a, (L,)), tl.reshape(b, (L,)), p)",
                "    else:",
                "        a0, a1 = tl.split(tl.permute(tl.reshape(a, (L, 2, K // 2)), (0, 2, 1)))",
                "        b0, b1 = tl.split(tl.permute(tl.reshape(b, (L, 2, K // 2)), (0, 2, 1)))",
                f"        p = {name}(p, a0, b0, L, K // 2)",
                f"        p = {name}(p, a1, b1, L, K // 2)",
                "    return p",
            ]
        else:
            add = self.ptx_helper("add.rn")
            lines = [
                "@triton.jit",
                f"def {name}(p, x, L: tl.constexpr, K: tl.constexpr):",
                "    # the chains p[j] = p[j] + x[j, k] for k = 0, 1, ..., K - 1 in order (note 21 of tilefish.lower): the columns of the low half, then of",
                "    # the high half, each half taken by a split (data movement); K is a power of two",
                "    if K == 1:",
                f"        p = {add}(p, tl.reshape(x, (L,)))",
                "    else:",
                "        x0, x1 = tl.split(tl.permute(tl.reshape(x, (L, 2, K // 2)), (0, 2, 1)))",
                f"        p = {name}(p, x0, L, K // 2)",
                f"        p = {name}(p, x1, L, K // 2)",
                "    return p",
            ]
        self.helpers[key] = (name, lines)
        return name

    def pad_helper(self) -> str:
        """Revision 2 (note 21): the helper that appends columns of one value (given by its bits: Triton 3.5.1 writes a literal ``-0.0`` as
        ``+0.0``) to a row (``tl.join``, a permute and a reshape: data movement) until it has ``W`` of them (``N`` and ``W`` powers of two)."""
        key = ("pad",)
        if key not in self.helpers:
            name = self.names.fresh("_tf_pad")
            self.helpers[key] = (name, [
                "@triton.jit",
                f"def {name}(x, bits: tl.constexpr, N: tl.constexpr, W: tl.constexpr):",
                "    # x followed by W - N columns of the f32 whose bits are the i32 bits (data movement; note 21 of tilefish.lower: a literal -0.0",
                "    # would be +0.0)",
                "    if N < W:",
                "        x = tl.reshape(tl.permute(tl.join(x, tl.full((N,), bits, tl.int32).to(tl.float32, bitcast=True)), (1, 0)), (2 * N,))",
                f"        x = {name}(x, bits, 2 * N, W)",
                "    return x",
            ])
        return self.helpers[key][0]

    def lane0_helper(self) -> str:
        """Revision 2 (note 21): the combine of ``(value, lane)`` pairs that keeps the lower lane: ``tl.reduce`` with it gives lane 0's value, bit
        for bit, whatever its tree (a select, and a total order on the lanes)."""
        key = ("lane0",)
        if key not in self.helpers:
            name = self.names.fresh("_tf_lane0")
            self.helpers[key] = (name, [
                "@triton.jit",
                f"def {name}(v1, i1, v2, i2):",
                "    # the pair of the lower lane, as it is (note 21 of tilefish.lower)",
                "    w = i1 < i2",
                "    return tl.where(w, v1, v2), tl.where(w, i1, i2)",
            ])
        return self.helpers[key][0]

    def split_sum_helper(self, order: Any, fused: bool) -> str:
        """Revision 2 of an ordered sum of ``lanes > 1`` (note 21): the columns past ``D`` made ``-0.0`` (``fma(-0.0, 1.0, acc)``), which leaves
        every value a chain can hold as it is, the row padded with such columns to whole rounds of ``lanes * vector``, reshaped to ``[lanes, K]``
        (lane ``j``'s columns in its order along the last axis), the chains taken by :meth:`chain_helper`, then the tree over the lanes, each
        ``p[j] + p[j + o]`` with ``tl.gather`` (data movement), and lane 0's value by :meth:`lane0_helper`. A lane of more than
        ``SPLIT_MAX_STEPS`` steps at a bucket takes the helper of revision 1 there (the same bytes)."""
        key = ("sum2", order.lanes, order.vector, order.offsets, fused)
        if key in self.helpers:
            return self.helpers[key][0]
        lanes, vector, width = order.lanes, order.vector, order.lanes * order.vector
        add = self.ptx_helper("add.rn")
        chain = self.chain_helper(fused)
        pad = self.pad_helper()
        lane0 = self.lane0_helper()
        fallback = self.gather_sum_helper(order, fused)
        stem = f"_tf_sum{'_fma' if fused else ''}_l{lanes}_v{vector}_o" + "_".join(map(str, order.offsets)) + "_split"
        name = self.names.fresh(stem)
        what = "fma(a, b, acc)" if fused else "acc + x"
        args = "a, b" if fused else "x"
        lines = [
            "@triton.jit",
            f"def {name}({args}, D: tl.constexpr, BLOCK: tl.constexpr):",
            f"    # an ordered sum (note 17 of tilefish.lower), {order}: lane j adds the columns c * {width} + j * {vector} + v in the order (c, v), each",
            f"    # step {what} rounded once, from +0.0; then p[j] = p[j] + p[j + o] for o = {', '.join(map(str, order.offsets))} (add.rn); the sum is lane 0's.",
            "    # Revision 2 (note 21): a column at or past D adds " + ("fma(-0.0, 1.0, acc)" if fused else "-0.0") + ", which leaves every value a chain holds as it is; the row",
            f"    # is padded with such columns to whole rounds and reshaped to [{lanes}, K], lane j's columns in order along the last axis; data movement",
            f"    P: tl.constexpr = BLOCK if BLOCK >= {width} else {width}",
            f"    if P // {lanes} <= {SPLIT_MAX_STEPS}:",
            "        cols = tl.arange(0, BLOCK)",
        ]
        neg0 = "tl.full((BLOCK,), -2147483648, tl.int32).to(tl.float32, bitcast=True)"  # -0.0 (a literal -0.0 would be +0.0)
        if fused:
            lines += [
                f"        a = {pad}(tl.where(cols < D, a, {neg0}), -2147483648, BLOCK, P)",
                f"        b = {pad}(tl.where(cols < D, b, 1.0), 1065353216, BLOCK, P)",
                f"        a = tl.reshape(tl.permute(tl.reshape(a, (P // {width}, {lanes}, {vector})), (1, 0, 2)), ({lanes}, P // {lanes}))",
                f"        b = tl.reshape(tl.permute(tl.reshape(b, (P // {width}, {lanes}, {vector})), (1, 0, 2)), ({lanes}, P // {lanes}))",
                f"        p = {chain}(tl.zeros(({lanes},), tl.float32), a, b, {lanes}, P // {lanes})",
            ]
        else:
            lines += [
                f"        x = {pad}(tl.where(cols < D, x, {neg0}), -2147483648, BLOCK, P)",
                f"        x = tl.reshape(tl.permute(tl.reshape(x, (P // {width}, {lanes}, {vector})), (1, 0, 2)), ({lanes}, P // {lanes}))",
                f"        p = {chain}(tl.zeros(({lanes},), tl.float32), x, {lanes}, P // {lanes})",
            ]
        lines.append(f"        lane = tl.arange(0, {lanes})")
        for o in order.offsets:  # every lane at once; a lane whose partner is past the end is not read again (only the live lanes reach lane 0)
            lines.append(f"        p = {add}(p, tl.gather(p, tl.minimum(lane + {o}, {lanes - 1}), 0))  # o = {o}: lanes j and j + {o}")
        lines += [
            f"        s, _ = tl.reduce((p, lane), 0, {lane0})",
            "    else:",
            f"        s = {fallback}({args}, D, BLOCK)  # a lane of more steps: revision 1's helper (the same order)",
            "    return s",
        ]
        self.helpers[key] = (name, lines)
        return name

    def gather_sum_helper(self, order: Any, fused: bool) -> str:
        """Revision 1 of an ordered sum (note 17): a loop over a lane's steps that takes each step's terms with ``tl.gather``, then the tree."""
        key = ("sum", order.lanes, order.vector, order.offsets, fused)
        if key in self.helpers:
            return self.helpers[key][0]
        lanes, vector, width = order.lanes, order.vector, order.lanes * order.vector
        add = self.ptx_helper("add.rn")
        step = self.ptx_helper("fma") if fused else ""
        first = self.first_helper()
        stem = f"_tf_sum{'_fma' if fused else ''}_seq" if lanes == 1 else f"_tf_sum{'_fma' if fused else ''}_l{lanes}_v{vector}_o" + "_".join(map(str, order.offsets))
        name = self.names.fresh(stem)
        columns = "the columns in order" if lanes == 1 else (
            f"lane j adds the columns c * {width} + j * {vector} + v, in the order (c, v)" if vector > 1 else f"lane j adds the columns c * {lanes} + j in order"
        )
        what = "fma(a, b, acc)" if fused else "acc + x"
        steps = "BLOCK" if width == 1 else f"(BLOCK + {width - 1}) // {width}" + (f" * {vector}" if vector > 1 else "")  # a lane's steps
        lines = [
            "@triton.jit",
            f"def {name}({'a, b' if fused else 'x'}, D: tl.constexpr, BLOCK: tl.constexpr):",
            f"    # an ordered sum (note 17 of tilefish.lower), {order}: {columns}, each step {what} rounded once, from +0.0;",
            "    # a column at or past D is skipped" + (
                "" if lanes == 1 else f"; then p[j] = p[j] + p[j + o] for o = {', '.join(map(str, order.offsets))} (add.rn); the sum is lane 0's"
            ),
            f"    lane = tl.arange(0, {lanes})",
            f"    p = tl.zeros(({lanes},), tl.float32)",
            f"    for s in range({steps}):",
        ]
        if vector == 1:
            lines.append(f"        col = s * {width} + lane" if lanes > 1 else "        col = s + lane")
        else:
            lines.append(f"        col = (s // {vector}) * {width} + lane * {vector} + s % {vector}")
        lines.append("        k = tl.minimum(col, BLOCK - 1)")
        if fused:
            lines.append(f"        p = tl.where(col < D, {step}(tl.gather(a, k, 0), tl.gather(b, k, 0), p), p)")
        else:
            lines.append(f"        p = tl.where(col < D, {add}(p, tl.gather(x, k, 0)), p)")
        n, removed = lanes, []
        for o in order.offsets:  # the live lanes, compacted: a lane's index without the bits of the offsets already taken
            q = o >> sum(1 for r in removed if r < o)
            lines.append(f"    a_, b_ = tl.split(tl.permute(tl.reshape(p, ({n // (2 * q)}, 2, {q})), (0, 2, 1)))  # o = {o}: lanes j and j + {o}")
            lines.append(f"    p = tl.reshape({add}(a_, b_), ({n // 2},))")
            n //= 2
            removed.append(o)
        lines.append(f"    return tl.reduce(p, 0, {first})")
        self.helpers[key] = (name, lines)
        return name

    def as_row(self, x: "Value | Scalar") -> str:
        """An operand of an fma sum as a ``[BLOCK_D]`` tensor: a per-row value or a scalar broadcast (exactly) along the row."""
        layout = self.layout
        assert isinstance(layout, _RowLayout)
        if isinstance(x, Value):
            if layout.cls[x] in (_FULL, _COL):
                return self.var[x]
            return f"tl.broadcast_to({self.var[x]}, ({layout.BLOCK_name},))"
        return f"tl.full(({layout.BLOCK_name},), {self.operand(x)}, tl.float32)"

    def ordered_sum(self, n: Node) -> str:
        """``sum(x, order=...)`` of an ``[R, D]`` value (note 17): the helper of its order on the row (on the operands of its product, ``fma=True``)."""
        layout = self.layout
        assert isinstance(layout, _RowLayout)
        x = n.inputs[0]
        assert isinstance(x, Value)
        fused = bool(n.attrs.get("fma"))
        if fused:
            product = self.producer[x]
            args = ", ".join(self.as_row(i) for i in product.inputs)
        else:
            args = self.var[x]
        return f"{self.sum_helper(n.attrs['order'], fused)}({args}, {layout.D_name}, {layout.BLOCK_name})"

    def f2i_helper(self, kind: str, rounding: str) -> str:
        """The name of the module-level helper of a float to integer conversion (note 18), written into :attr:`helpers` the first time."""
        key = ("f2i", kind, rounding)
        if key not in self.helpers:
            name = self.names.fresh(f"_tf_f2i_{kind}_{rounding}")
            doc, body = _f2i_helper(kind, rounding)
            self.helpers[key] = (name, ["@triton.jit", f"def {name}(x):", *(f"    # {line}" for line in doc), *(f"    {line}" for line in body)])
        return self.helpers[key][0]

    def float_to_int(self, n: Node, a: str, v: str) -> str:
        """A float to integer cast with its declared rounding and saturation (note 18): a narrower float is widened to f32 first (exact), then
        one helper of inline PTX. Saturating: ``cvt`` to ``s32`` (``i32``), to ``s32`` clamped to ``[0, 255]`` (``u8``) or to ``s64`` with a NaN
        made 0 (``i64``). Wrapping: the ``s64`` helper, then the low bits (``.to()`` of an integer to a narrower one truncates)."""
        x = n.inputs[0]
        assert isinstance(x, Value)
        dst, rounding, saturate = n.attrs["to"].name, n.attrs["round"], n.attrs["saturate"]
        if x.type.dtype.name != "f32":
            wide = self.names.fresh(f"{v}_x")
            self.emit(f"{wide} = {a}.to(tl.float32)", "exact")
            a = wide
        if dst == "i32" and saturate:
            return f"{self.f2i_helper('s32', rounding)}({a})"
        if dst == "u8" and saturate:
            return f"{self.f2i_helper('u8', rounding)}({a}).to(tl.uint8)"  # in [0, 255]: the low byte is the value
        wide64 = f"{self.f2i_helper('s64', rounding)}({a})"
        if dst == "i64":  # saturating and wrapping are one for i64
            return wide64
        return f"{wide64}.to(tl.{_TL[dst]})"  # wrapping: the low bits of the i64, modulo 2**bits

    def arg_lanes(self, n: Node, v: str) -> "tuple[str, str]":
        """The (value, index) pair each lane of the row (or of the chunk) offers an argmax / argmin; a masked lane loses to every lane."""
        layout = self.layout
        assert isinstance(layout, _RowLayout)
        x = n.inputs[0]
        assert isinstance(x, Value)
        a = self.var[x]
        if x.type.dtype.name in ("bf16", "f16"):  # widened to f32, exactly
            wide = self.names.fresh(f"{v}_x")
            self.emit(f"{wide} = {a}.to(tl.float32)", "exact")
            a = wide
        _, sentinel, _ = self.arg_kind(n)
        return f"tl.where({layout.mask}, {a}, {sentinel})", f"tl.where({layout.mask}, {layout.cols}, {_INDEX_SENTINEL})"

    def arg_reduce(self, n: Node, v: str) -> str:
        """``argmax`` / ``argmin`` of an ``[R, D]`` value along the row in one block (note 8): one ``tl.reduce`` of the (value, index) pairs
        with the combine of the strict total order, so the result depends on no tree."""
        value, index = self.arg_lanes(n, v)
        self.emit(f"{v}_v, {v}_i = tl.reduce(({value}, {index}), 0, {self.combine(n)})")
        out = n.outputs[0].type.dtype.name
        return f"{v}_i" if out == "i32" else f"{v}_i.to(tl.{_TL[out]})"

    def expression(self, n: Node, v: str = "") -> str:
        op = n.op
        if op in _ARG_REDUCTIONS:
            return self.arg_reduce(n, v)
        if op == "sum" and "order" in n.attrs:  # note 17 (an fma sum reads its product's operands, not the product)
            return self.ordered_sum(n)
        if op == "mean" and "order" in n.attrs:  # note 21: the ordered sum times f32(1 / D), one mul.rn
            layout = self.layout
            assert isinstance(layout, _RowLayout)
            return f"{self.ptx_helper('mul.rn.f32')}({self.ordered_sum(n)}, 1.0 / {layout.D_name})"
        if op == "gather_rows":
            return self.gather(n, v)
        if op == "load":  # one element from device memory, broadcast: no mask, no offsets
            ptr = n.inputs[0]
            assert isinstance(ptr, Value)
            return f"tl.load({self.rename.get(ptr.name, ptr.name)})"
        if op in _COMPARE:
            return f"{self.typed(n.inputs[0])} {_COMPARE[op]} {self.typed(n.inputs[1])}"
        if op == "where":
            return f"tl.where({self.operand(n.inputs[0])}, {self.typed(n.inputs[1])}, {self.typed(n.inputs[2])})"
        if op == "iota":
            index = self.layout.iota(n.outputs[0])
            return index if n.outputs[0].type.dtype.name == "i32" else f"{index}.to(tl.{_TL[n.outputs[0].type.dtype.name]})"
        if op in ("floordiv", "mod"):
            return self.int_division(n, v)
        a = [self.operand(i) for i in n.inputs]
        if op in _LOGICAL:
            return f"{a[0]} {_LOGICAL[op]} {a[1]}"
        if op == "not":
            return f"~{a[0]}"
        if op == "tanh":
            self.libdevice = True  # libdevice's tanhf (note 14): the module imports it
        if self.math_of(n) == "ieee" and op in _IEEE:  # note 13: CUDA's f32 math without fast math or flush to zero
            return f"{self.ieee_helper(op)}({', '.join(a)})"
        if op == "tanh":
            return f"libdevice.tanh({a[0]})"
        if op == "erf":
            return f"tl.math.erf({a[0]})"
        if op in ("sin_approx", "rcp", "fma"):  # one PTX instruction in either math (note 16)
            return f"{self.ptx_helper(op)}({', '.join(a)})"
        if op == "neg":
            dt = n.outputs[0].type.dtype
            if dt.is_float:  # CUDA's neg, its NaN the canonical one whatever ptxas folds the negation into (note 16)
                return f"{self.ptx_helper('negc.' + dt.name)}({a[0]})"
            return f"-{a[0]}"  # an integer: 0 - x, two's complement
        rn = self.strict or n in self.rn_products  # separate roundings (notes 2 and 17): one scalar .rn instruction, never packed or contracted
        if op in _BINARY and rn and op != "div" and n.outputs[0].type.dtype.name in _RN:
            return f"{self.ptx_helper(op + '.rn.' + n.outputs[0].type.dtype.name)}({self.typed(n.inputs[0])}, {self.typed(n.inputs[1])})"
        if op in _BINARY:
            return f"{a[0]} {_BINARY[op]} {a[1]}"
        if op == "rsqrt":
            return f"tl.rsqrt({a[0]})"
        if op == "exp":
            return f"tl.exp({a[0]})"
        if op == "sqrt":
            return f"tl.sqrt_rn({a[0]})"
        if op in ("sin", "cos"):
            return f"tl.{op}({a[0]})"
        if op == "silu" and self.strict:  # notes 2 and 20: its last multiply one rounding, never contracted with what reads it
            return f"{self.ptx_helper('mul.rn.f32')}({a[0]}, tl.sigmoid({a[0]}))"
        if op == "silu":
            return f"{a[0]} * tl.sigmoid({a[0]})"
        if op == "gelu" and self.strict:  # the same evaluation, every multiply and add one rounding (notes 2 and 20)
            mul, add = self.ptx_helper("mul.rn.f32"), self.ptx_helper("add.rn.f32")
            return f"{mul}({mul}(0.5, {a[0]}), {add}(1.0, tl.math.erf({mul}({a[0]}, 0.7071067811865476))))"
        if op == "gelu":
            return f"0.5 * {a[0]} * (1.0 + tl.math.erf({a[0]} * 0.7071067811865476))"
        if op == "cast":
            if "round" in n.attrs:  # float to integer, the declared rounding and saturation (note 18)
                return self.float_to_int(n, a[0], v)
            if n.attrs.get("nan") == "preserve":  # f32 -> bf16 by the integer round to nearest even on the bits (note 1)
                b, r, q = (self.names.fresh(f"{v}_{s}") for s in ("b", "r", "q"))
                self.emit(f"{b} = {a[0]}.to(tl.uint32, bitcast=True)")
                self.emit(f"{r} = ({b} + 0x7FFF + (({b} >> 16) & 1)) >> 16", "round to nearest even on the upper half")
                self.emit(f"{q} = ({b} >> 16) | 0x40", "a NaN: its sign and payload, quiet")
                return f"tl.where({a[0]} != {a[0]}, {q}, {r}).to(tl.uint16).to(tl.bfloat16, bitcast=True)"
            dst = n.attrs["to"].name
            src = n.inputs[0].type.dtype.name  # type: ignore[union-attr]
            narrowing = n.attrs.get("rounding") and dst in ("bf16", "f16") and src in ("f32", "bf16", "f16")
            if narrowing and (self.strict or self.widened(n.inputs[0])):  # inline cvt, out of the reach of the folds (notes 1, 2 and 20)
                x = a[0]
                if src != "f32":
                    x = self.names.fresh(f"{v}_x")
                    self.emit(f"{x} = {a[0]}.to(tl.float32)", "exact")
                return f"{self.ptx_helper('cvt.' + dst)}({x})"
            if self.lowering >= 2 and self.strict and dst in ("e4m3", "e5m2") and src in ("f32", "bf16", "f16"):  # note 21: an inline fp8 cvt
                x = a[0]
                if src != "f32":
                    x = self.names.fresh(f"{v}_x")
                    self.emit(f"{x} = {a[0]}.to(tl.float32)", "exact")
                return f"{self.ptx_helper('cvt.' + dst)}({x}).to(tl.uint8).to(tl.{_TL[dst]}, bitcast=True)"  # the low byte, as it is
            return f"{a[0]}.to(tl.{_TL[dst]})"
        if op in _REDUCTIONS:
            return self.layout.reduce(op, a[0])
        raise AssertionError(f"unreachable op {op}")  # pragma: no cover - _check_common rejects it


# -- assembly -------------------------------------------------------------------------------------


def _stub(fn_name: str, module: str) -> Any:
    def stub() -> None: ...

    stub.__name__ = stub.__qualname__ = fn_name
    stub.__module__ = module
    stub.__doc__ = f"The generated Triton kernel {module}:{fn_name}."
    return stub


def _check_names(op: Any, module: "str | None") -> "tuple[str, str]":
    """Validate the kernel and the module name; return ``(module, function name)``."""
    if not isinstance(op, KernelOp):
        raise TilefishLoweringError(
            f"lower takes a @tilefish.kernel, got {op!r}; only kernels have a body to lower", rule="not-a-kernel"
        )
    own = op.base.name if op.base is not None else op.name  # a specialisation for unaligned pointers (KernelOp.unaligned) has its kernel's source
    fn_name = f"tf_{own}"
    if not _IDENT.match(fn_name):
        raise _fail(op.name, None, "name", f"'{fn_name}' is not a plain ASCII function name; give the kernel a name of letters, digits and underscores")
    if module is None:
        module = f"{DEFAULT_MODULE}.{own}"
    if not isinstance(module, str) or not _MODULE.match(module) or any(keyword.iskeyword(p) for p in module.split(".")):
        raise TilefishLoweringError(f"lower: module={module!r} must be a dotted Python module name", rule="module")
    return module, fn_name


def lower(
    op: KernelOp,
    *,
    module: "str | None" = None,
    template: "str | None" = None,
    num_warps: "int | None" = None,
    fp_fusion: "bool | None" = None,
    math: "str | None" = None,
    context_free: "bool | None" = None,
    lowering: "int | None" = None,
) -> LoweredKernel:
    """Lower a ``@tilefish.kernel`` to Triton source.

    ``module`` is the Python module the generated source will live in (default
    ``"tilefish_generated.<kernel name>"``); the Triton reference is ``module:tf_<name>``.
    ``template`` forces ``"row"`` or ``"elementwise"``; by default the elementwise template is
    tried first and then the row template. ``num_warps`` overrides the kernel's own
    ``num_warps=`` and the template's rule, and pins the warp count of every bucket; without either,
    each bucket of a bucketed ``D`` gets the warps of its own size. ``fp_fusion`` overrides the kernel's
    own ``fp_fusion=`` (``False``: no FMA contraction), and ``math`` its ``math=`` (``"ieee"``: note 13; an op
    that says its own keeps it). ``context_free`` overrides the kernel's own (``True``: the code of ``fp_fusion=False``, every
    float ``+ - *`` and narrowing cast inline PTX, and views read where they are, note 20; what a graph's fused regions are lowered with).
    ``lowering`` overrides the kernel's own revision of the lowering (``1``, every kernel's; ``2``, what the regions of ``tilefish.fusion(version=2)``
    are lowered with: ordered sums by halving splits, the warp rule of note 21 and, context-free, views read where they are by the row template;
    the bytes of revision 1 but a ``-0.0`` scalar's sign, other code). Raises :class:`~tilefish.errors.TilefishLoweringError` when the body fits no template.
    """
    module, fn_name = _check_names(op, module)
    if template is not None and template not in TEMPLATES:
        raise TilefishLoweringError(f"lower: template must be 'row' or 'elementwise', got {template!r}", rule="template")
    if fp_fusion is None:
        fp_fusion = op.fp_fusion
    if not isinstance(fp_fusion, bool):
        raise TilefishLoweringError(f"lower: fp_fusion must be True or False, got {fp_fusion!r}", rule="fp-fusion")
    if math is None:
        math = getattr(op, "math", "fast")
    if math not in ("fast", "ieee"):
        raise TilefishLoweringError(f"lower: math must be 'fast' or 'ieee', got {math!r}", rule="math")
    if context_free is None:
        context_free = bool(getattr(op, "context_free", False))
    if not isinstance(context_free, bool):
        raise TilefishLoweringError(f"lower: context_free must be True or False, got {context_free!r}", rule="context-free")
    if lowering is None:
        lowering = getattr(op, "lowering", 1)
    if isinstance(lowering, bool) or lowering not in LOWERINGS:
        raise TilefishLoweringError(f"lower: lowering must be one of {list(LOWERINGS)}, got {lowering!r}", rule="lowering")
    if num_warps is None:
        num_warps = op.num_warps
    if num_warps is not None and (isinstance(num_warps, bool) or not isinstance(num_warps, int) or num_warps not in NUM_WARPS):
        raise TilefishLoweringError(
            f"lower: num_warps must be a power of two from 1 to 32, got {num_warps!r}", rule="num-warps"
        )
    _check_common(op)
    tables = _tables(op)
    # without a template: the elementwise template with operands of one shape, then the row template, then the elementwise template
    # broadcasting an operand over the leading axes (a last resort, so that no kernel the row template takes changes template)
    attempts = {"row": [("row", False)], "elementwise": [("elementwise", True)]}.get(
        template or "", [("elementwise", False), ("row", False), ("elementwise", True)]
    )
    failed: list[tuple[str, TilefishLoweringError]] = []
    for t, bcast in attempts:
        try:
            layout = _ElementwiseLayout(op, tables, broadcast=bcast) if t == "elementwise" else _RowLayout(op, tables, lowering)
        except TilefishLoweringError as e:
            if not (t == "elementwise" and not bcast):  # the broadcasting attempt says more
                failed.append((t, e))
            continue
        return _assemble(op, layout, module, fn_name, num_warps, fp_fusion, math, context_free, lowering)
    errors = [e for _, e in sorted(failed, key=lambda f: TEMPLATES.index(f[0]))]
    if len(errors) == 1:
        raise errors[0]
    detail = "; ".join(f"{e.rule}: {str(e).split('failed: ', 1)[-1]}" for e in errors)
    raise TilefishLoweringError(
        f"cannot lower kernel '{op.name}': it fits no template. {detail}", rule="no-template", causes=tuple(errors)
    )


def _assemble(
    op: KernelOp,
    layout: _Layout,
    module: str,
    fn_name: str,
    num_warps: "int | None",
    fp_fusion: bool,
    math: str = "fast",
    context_free: bool = False,
    lowering: int = 1,
) -> LoweredKernel:
    graph = op.graph
    user = list(op.signature.params)
    names = _Names({p.name for p in user} | _RESERVED)
    # a kernel parameter that has the name of something the generated module uses would shadow it
    rename = {p.name: names.fresh(p.name) for p in user if p.name in _RESERVED}
    out_names = (
        [names.fresh("out")] if len(graph.outputs) == 1 else [names.fresh(f"out{k}") for k in range(len(graph.outputs))]
    )
    taken_dims: set[str] = set()
    for p in user:
        if p.kind == "tensor":
            assert p.type is not None
            for d in p.type.shape:
                taken_dims.update(d.named_dims())
        elif p.kind == "dim":
            assert p.dim is not None
            taken_dims.update(p.dim.named_dims())
    extras = layout.extras(names, taken_dims)
    bucket_warps: dict[str, int] = {}
    if num_warps is None:
        num_warps = layout.default_num_warps()  # for a bucketed D: the largest bucket's, which no binary of a bucket falls back to
        bucket_warps = layout.bucket_num_warps()  # each bucket has the warps of its own BLOCK_D
    warps_header = (
        f"num_warps: {num_warps}"
        if not bucket_warps
        else "num_warps: " + ", ".join(f"{w} for {s.lstrip('.')}" for s, w in bucket_warps.items())
    )

    # the parameters of the generated function and of its typed stub, in order
    decls: list[str] = []
    params: list[Param] = []
    max_elements: dict[str, int] = {}  # tensors whose size no static bound keeps within i32 offsets
    for p in user:
        new = rename.get(p.name, p.name)
        # (a parameter of a specialisation for unaligned pointers, KernelOp.unaligned, is compiled aligned to its element only)
        params.append(_reindexed(p, len(params), new, unaligned=p.name in op.unaligned_params))
        decls.append(f"{new}: tl.constexpr" if p.kind == "const" else new)
        if p.kind == "tensor" and (p.strided or _numel_bounds(p.type.shape)[1] is None):  # type: ignore[union-attr]
            max_elements[new] = INDEX_MAX  # (a view: its largest offset, max_extent)
    for k, v in enumerate(graph.outputs):
        params.append(Param(out_names[k], len(params), "tensor", "out", type=v.type))
        decls.append(out_names[k])
        if _numel_bounds(v.type.shape)[1] is None:
            max_elements[out_names[k]] = INDEX_MAX
    for e in extras:
        params.append(_reindexed(e.param, len(params), e.param.name))
        decls.append(f"{e.name}: tl.constexpr" if e.constexpr else e.name)
    element_counts: dict[str, str] = {}
    if layout.template == "elementwise":  # n is the element count of every tensor of the kernel's shape: say which one a host reads it from
        assert isinstance(layout, _ElementwiseLayout)
        tables = {rename.get(t.name, t.name) for t in layout.tables.only}
        element_counts[layout.n_name] = next(
            p.name
            for p in params
            if p.kind == "tensor" and not p.type.devscalar and p.name not in tables and p.type.shape == layout.shape  # type: ignore[union-attr]
        )
        element_counts.update({k: rename.get(v, v) for k, v in layout.counts.items()})  # a broadcast operand's own count

    body_emitter = _Body(op, layout, names, out_names, rename, math, context_free, fp_fusion, lowering)
    body = body_emitter.run()
    header = [
        f"Generated by tilefish.lower from the kernel '{op.base.name if op.base is not None else op.name}'. Do not edit.",  # (a specialisation has its kernel's source)
        *layout.header(),
        warps_header,
        *(
            ["fp_fusion: False (no FMA contraction: a * b + c rounds twice; every float + - * is one scalar .rn instruction, every narrowing cast inline PTX: note 2)"]
            if not fp_fusion
            else []
        ),
        *([_IEEE_HEADER] if math == "ieee" else []),
        *(["context-free: every narrowing float cast is inline PTX, a negation's NaN canonical (note 20: a fused region of a graph)"] if context_free else []),
        *(
            ["lowering: revision 2 (note 21: ordered sums by halving splits, the warp rule of rows with ordered sums, views read where they are by the row template)"]
            if lowering >= 2
            else []
        ),
        *[f"parameter '{old}' of the kernel is '{new}' here (the name is used by this module)" for old, new in rename.items()],
        "Numerics contract: see the docstring of tilefish.lower.",
    ]
    lines = [f"# {h}" for h in header]
    lines += ["import triton", "import triton.language as tl"]
    if body_emitter.libdevice:  # tanh: libdevice by name (note 14)
        lines.append("from triton.language.extra import libdevice")
    helper_spans: "list[tuple[int, int, str]]" = []  # the lines of each helper function (0-based, end excluded), and its name
    for _, helper in body_emitter.helpers.values():
        lines += ["", ""]
        start = _text_lines(lines)
        lines += helper
        named = re.search(r"^def (\w+)\(", "\n".join(helper), re.M)
        helper_spans.append((start, _text_lines(lines), named.group(1) if named else "helper"))
    lines += ["", "", "@triton.jit", f"def {fn_name}("]
    lines += [f"    {d}," for d in decls]
    lines += ["):"]
    first = _text_lines(lines)
    lines += body
    source = "\n".join(lines) + "\n"
    line_map = _line_map(op, first, body_emitter, source, helper_spans)

    extern = ExternOp(
        _stub(fn_name, module),
        op.name,
        Signature(params, ()),
        f"{module}:{fn_name}",
        num_warps,
        NUM_STAGES,
        layout.grid,
        max_elements=max_elements,
        element_counts=element_counts,
        fp_fusion=fp_fusion,
        bucket_num_warps=bucket_warps,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        sass=getattr(op, "sass", None),
    )
    return LoweredKernel(op.name, fn_name, module, layout.template, source, extern, num_warps, op, line_map)


def _text_lines(elements: "list[str]") -> int:
    """The lines of text ``"\\n".join(elements)`` has (an element may hold several)."""
    return sum(e.count("\n") + 1 for e in elements)


def _line_map(op: KernelOp, first: int, body: "_Body", source: str, helpers: "Sequence[tuple[int, int, str]]" = ()) -> "tuple[Any, ...]":
    """What each line of ``source`` was written for (:attr:`LoweredKernel.line_map`): the body's lines start after ``first`` lines (a line of the
    body may hold several lines of text); the lines of a helper function (``helpers``: its span and name) were written for the ops whose lines
    call it (one: that op, with ``"helper"``; several: ``{"what": "helper <name>", "used_by": [...]}``)."""
    index = {id(n): k for k, n in enumerate(op.graph.nodes)}

    def entry_of(made: Any) -> Any:
        if isinstance(made, Node):
            entry: "dict[str, Any]" = {"node": index.get(id(made)), "op": made.op}
            if made.loc is not None:
                entry["loc"] = f"{made.loc[0]}:{made.loc[1]}"
            if made.origin is not None:
                entry["origin"] = made.origin
            return entry
        return {"what": made} if isinstance(made, str) else None

    out: "list[Any]" = [None] * first
    users: "dict[str, list[Any]]" = {name: [] for _, _, name in helpers}
    for text, made in zip(body.lines, body.made_for):
        entry = entry_of(made)
        out += [entry] * (text.count("\n") + 1)
        if isinstance(made, Node):
            for name in users:
                if re.search(rf"(?<![\w.]){re.escape(name)}\(", text) and entry not in users[name]:
                    users[name].append(entry)
    for start, end, name in helpers:
        found = users[name]
        if len(found) == 1:
            label: Any = {**found[0], "helper": name}
        else:
            ops = ", ".join(f"{u['node']} {u['op']}" for u in found)
            label = {"what": f"helper {name}" + (f" (for nodes {ops})" if ops else ""), "used_by": found}
        for k in range(start, min(end, len(out))):
            out[k] = label
    total = source.count("\n")
    return tuple(out[:total] + [None] * (total - len(out)))


def _reindexed(p: Param, index: int, name: str, unaligned: bool = False) -> Param:
    return Param(
        name,
        index,
        p.kind,
        p.access,
        p.type,
        p.const_dtype,
        p.default,
        p.dim,
        p.keyword_only,
        strided=p.strided,  # (a fused region's view input, note 20: its strides are its StrideOf parameters)
        align=p.align,
        stride_of=p.stride_of,
        stride_dtype=p.stride_dtype,
        unaligned=unaligned,
    )
