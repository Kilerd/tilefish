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
``scatter_rows`` (``[N, D]`` or ``[N]``, addressed by an index; note 9). Reductions (``sum``, ``mean``,
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
``n = prod(shape)`` (but an ``[N]`` table of ``gather_rows`` / ``scatter_rows``, addressed by an index);
no broadcasting except python and ``const`` scalars; no reductions. ``n`` is a
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
lowered: rule ``view`` or ``split`` says so, and a strided view goes to an ``extern`` with a ``Strided[...]`` parameter.

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
``lower(fp_fusion=False)``, which compiles without FMA contraction (note 2). The key is part of the
declaration digest and of what ``verify_against`` checks; a hand-written ``extern`` never has it.

Parameters of the generated function, in this order: the kernel's own parameters under their own
names (tensors are pointers, ``const`` parameters are ``tl.constexpr`` with the declared default
as the spec's constant, runtime dimension parameters are ``i32``), one output pointer per returned
value (``out``, or ``out0``, ``out1``, ... for several), then the template's parameters. A
generated name that collides with a kernel parameter is suffixed (``mask_1``). Constexpr values come
from the spec (``compile_specs(lowered.extern)``), not from defaults in the source.

A kernel parameter whose name the generated module itself uses (``tl``, ``triton``, ``float``) or
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
                                                                                         A float to integer cast is rejected
                                                                                         (rule ``cast.float-to-int``).
                                                                                         A narrowing cast gets ``# rounding
                                                                                         point``. Nothing else is ever cast.
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
                                                                                         one FMA (note 2).
``a / b``                   ``a / b``                                                       f32 only (note 4): ``div.full.f32``,
                                                                                         not IEEE division.
``rsqrt(x)``                ``tl.rsqrt(x)``                                                 f32 only; approximate (note 5).
``exp(x)``                  ``tl.exp(x)``                                                   f32 only; fast exp (note 5).
``sqrt(x)``                 ``tl.sqrt_rn(x)``                                               f32 only; IEEE round to nearest, but
                                                                                         subnormals flush to zero (note 5).
``silu(x)``                 ``x * tl.sigmoid(x)``                                           f32 only; ``tl.sigmoid`` is
                                                                                         ``1 / (1 + tl.exp(-x))``.
``gelu(x)``                 ``0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))``       f32 only; exact erf form, evaluated
                                                                                         left to right (note 5).
``sin(x)``, ``cos(x)``      ``tl.sin(x)``, ``tl.cos(x)``                                    f32 only; libdevice ``__nv_sinf`` /
                                                                                         ``__nv_cosf``, built with FTZ: torch's
                                                                                         CUDA bits except ``sin`` of a
                                                                                         subnormal, which is ±0 (note 10).
``sum(x, axis=-1)``         ``tl.sum(tl.where(mask, x, 0.0), axis=0)``                      f32 only. The reduction is over all
                                                                                         ``BLOCK_D`` lanes; masked lanes are
                                                                                         set to 0 explicitly (intermediates
                                                                                         such as ``exp(x)`` or ``x - mean``
                                                                                         are not 0 there). The order is
                                                                                         Triton's, and depends on
                                                                                         ``num_warps`` (note 6).
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
``argmax(x, -1, dtype=)``   row template only: ``tl.max`` of the row (``-inf`` in the       exact and order-free: the first NaN,
``argmin(x, -1, dtype=)``   masked lanes), then ``tl.min`` of ``cols`` where the row        else the first extreme value
                            holds it, or where it holds a NaN (note 8)                      (``-0.0 == +0.0``); torch's rule.
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
   casts are the plain conversions. A **float to integer cast is rejected** by the lowering (rule
   ``cast.float-to-int``), because its rounding and saturation modes are not supported yet; what Triton does,
   measured for a hand-written kernel on sm_103, is the reason: ``i32`` is ``cvt.rzi.s32.f32`` (truncation toward zero,
   saturating, NaN is 0), ``i64`` is ``cvt.rzi.s64.f32`` (the same, but NaN is ``-2**63``), and ``u8`` is
   ``cvt.rzi.u16.f32`` followed by keeping the low byte: truncated and clamped to [0, 65535] (negative and NaN
   give 0), then reduced modulo 256, so ``-1.5`` is 0, ``255.5`` is 255, ``256`` is 0, ``300`` is 44, ``1000`` is
   232 and anything from 65535 up is 255; it neither saturates at 255 nor wraps like C. torch differs again
   (``-1.5`` to ``u8`` is 255).
2. **FMA contraction.** A generated kernel asks Triton for ``enable_fp_fusion=True`` (Triton's default, written
   into the spec), so ``a * b + c`` **may round once**: it is one FMA, ``fma.rn.f32`` for f32 and ``fma.rn.bf16x2``,
   a native packed bf16 FMA, for bf16. With ``@tilefish.kernel(fp_fusion=False)`` it is ``mul.rn`` then ``add.rn``
   (two roundings, "strict"). The two results differ in 23% of random elements (22.9% in bf16, 23.4% in f32), and
   the contracted one is exactly the correctly rounded value of the exact product and sum. For the tail of layernorm,
   ``(x - mean) * rstd * w + b`` in f32, contraction changes 25.7% of the elements and the strict build matches a
   reference that rounds every operation exactly. On the goldens contraction changes the bytes of the ones that
   have a contractable ``a * b + c`` (layernorm: 3 to 8 of 197376 elements, one bf16 ULP; the README rmsnorm: 1 to 2
   of 657920); softmax, gelu and swiglu come out byte-identical either way. The explicit FMAs inside libdevice's
   ``erf`` stay with ``fp_fusion=False``.
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
   wins. So the first NaN wins, then the first extreme value; a row of ``-inf`` gives 0 for ``argmax``. This is torch's rule
   (``torch.argmax`` / ``torch.argmin`` on CUDA); it is not ``tl.argmax``'s, whose combine lets a NaN win only from one side, so its
   answer depends on the tree. A bf16 or f16 value is widened to f32 (exact), an integer is compared as it is (``u8`` unsigned). The
   lowering is three or four reductions over the row, every one a ``tl.max`` or ``tl.min`` (whether there is a NaN; the extreme over the
   valid lanes, the masked lanes being ``-inf`` / ``+inf`` or the integer type's bound; the smallest ``cols`` that holds the NaN or the
   extreme), and a max or a min is exact and independent of the reduction tree: unlike ``sum``, the result does not depend on
   ``num_warps`` (whatever ``tl.max`` does with a NaN or with the two zeros, the lanes are then picked by equality and the NaN test).
   Only the row template has it, along the last axis.
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

``num_warps`` is part of a kernel's numerics identity. It fixes the tree of ``tl.sum`` and ``tl.max`` (note 6), so the same
kernel with another warp count is another numerics: its result differs in the last bits, like a changed reduction order or
accumulator dtype (a numerics-changing rewrite in the README's terms), and two binaries that differ in ``num_warps`` are not
byte-comparable. It is therefore recorded everywhere an identity is: the declaration digest, the compile spec, the manifest
entry (a replay refuses an entry whose ``num_warps`` is not the spec's) and the manifest digest. Pin it
(``@tilefish.kernel(num_warps=N)``) where bytes must not move when a size is added to a bucketed dimension, and treat a
change of it, or of the rule that picks it (``clamp(BLOCK_D // 256, 1, 16)``), as a change of the numerics contract.

The ``reference`` mode is held to the binary within tolerances that are measured per input distribution, not stated once:
a ULP count only means something where the result is far from zero, so for a bf16 output the bound is half a bf16 step of the
largest value of the reference (``2**-7 * max|ref|``) together with a fraction of elements allowed to differ, for the f32 softmax an
absolute error of 5e-7 (and ULP where meaningful), for the gelu an absolute error of 3e-5. They differ by dtype, kernel and
distribution (the mean of a layernorm's data, the scale of a softmax's), and the measured bounds are tabulated, and asserted, in
``tests/test_gpu_reference.py``.
"""

from __future__ import annotations

import hashlib
import itertools
import keyword
import math
import pathlib
import re
import struct
from dataclasses import dataclass
from typing import Any, Iterable

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
MAX_BLOCK = 1 << 20  # Triton's limit on the number of elements of a tensor
NUM_STAGES = 3
NUM_WARPS = (1, 2, 4, 8, 16, 32)
DEFAULT_MODULE = "tilefish_generated"
TEMPLATES = ("elementwise", "row")
INDEX_MAX = MAX_ELEMENTS  # offsets are i32: a tensor has at most 2**31 - 1 elements
# Names a kernel parameter cannot keep in the generated function: what the module itself uses, and what
# Triton's JIT launcher takes as launch options (a parameter of that name could not be passed by keyword).
_RESERVED = frozenset({"tl", "triton", "float", "grid", "warmup", "debug", "num_warps", "num_stages", "num_ctas", "maxnreg"})

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
_UNARY = ("rsqrt", "exp", "sqrt", "silu", "gelu", "sin", "cos")
_REDUCTIONS = ("sum", "mean", "max")
_ARG_REDUCTIONS = ("argmax", "argmin")
# Triton's own type of a python literal: a scalar of any other dtype is written as a typed constant (tl.cast(0.1, tl.bfloat16)),
# because Triton compares a bf16 or f16 tensor with a float literal in f32 (the literal is not rounded to the tensor's dtype first)
_LITERAL_DTYPES = frozenset({"f32", "i32"})

_FULL, _COL, _ROW = "full", "col", "row"  # row template: [R, D], [D], [R] / [R, 1]
_SCALAR, _DEV = "scalar", "devscalar"  # rank 0: a value read from a DevScalar (broadcast), and the DevScalar parameter itself (a pointer to one element)


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
    for v in graph.inputs:
        if not v.type.contiguous:
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
                "where, iota, silu, gelu, rsqrt, exp, sqrt, sin, cos, .to, sum, mean, max, argmax, argmin, gather_rows, scatter_rows, load and "
                "stores) are lowered",
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
        if n.op in ("reshape", "permute", "select"):
            raise _fail(
                name,
                None,
                "view",
                f"{n} is a view operation, and strided views are not lowered: a lowered kernel takes and gives contiguous tensors. "
                "Write the kernel over the tensor the view is of, or declare an extern with a Strided[...] parameter",
            )
        for v in n.outputs:
            if not v.type.contiguous:
                raise _fail(name, None, "view-value", f"{_label(v)} is a View; strided views are not lowered")
        _check_node_dtypes(name, n)
    for v in graph.outputs:
        if not v.type.contiguous:
            raise _fail(name, None, "view-value", f"the returned {_label(v)} is a View; strided views are not lowered")
    _check_index_range(name, _all_values(graph))


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
    elif n.op in _UNARY:
        dt = n.outputs[0].type.dtype
        if dt.name not in _MATH:
            raise _fail(
                kernel,
                None,
                "math-dtype",
                f"{n}: {n.op} of a {dt} tensor is not lowered (Triton's math functions are f32 here); cast with .to(f32) first",
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
        if src.is_float and dst.is_int:
            raise _fail(
                kernel,
                None,
                "cast.float-to-int",
                f"{n}: casting {src} to {dst} is not lowered: the rounding and saturation modes of a float to integer "
                "cast are not supported yet (Triton truncates toward zero and, out of range, saturates or keeps only "
                "the low byte, depending on the integer type; see note 1 of the numerics contract). Round or clamp "
                "explicitly in a typed op once one exists, or declare the kernel with extern",
            )
    elif n.op == "load":
        _check_dtype(kernel, f"the DevScalar read by {n}", n.outputs[0].type.dtype)
    elif n.op in _COMPARE or n.op in _LOGICAL or n.op in ("not", "where", "iota", "gather_rows", "scatter_rows") or n.op in _ARG_REDUCTIONS:
        for v in (*n.outputs, *(i for i in n.inputs if isinstance(i, Value))):
            _check_dtype(kernel, f"{v.ref} of {n}", v.type.dtype)
    elif n.op != "store":
        raise _fail(kernel, None, "unsupported-op", f"the op '{n.op}' is not lowered")


# -- tables: the parameters a gather reads or a scatter writes at computed rows ---------------------------------------------------


@dataclass
class _Tables:
    """The parameters that ``gather_rows`` reads and ``scatter_rows`` writes at computed rows (note 9).

    ``params`` maps each to ``"gather"`` or ``"scatter"``, in the order the kernel declares them; ``only`` are those that nothing else in the
    body reads (the prologue does not load them and the template does not fit them to its shapes: their rows are addressed by the index);
    ``versions`` are the versions of the targets that the scatters make (written in place, read by nothing).
    """

    params: "dict[Value, str]"
    only: "set[Value]"
    versions: "set[Value]"

    def skip(self, v: Value) -> bool:
        """Whether the template leaves ``v`` out of its shapes: a table that only the index addresses, or the version a scatter makes."""
        return v in self.only or v in self.versions


def _tables(op: KernelOp) -> _Tables:
    graph, name = op.graph, op.name
    found: dict[Value, str] = {}
    versions: set[Value] = set()
    for n in graph.nodes:
        if n.op == "gather_rows":
            x = n.inputs[0]
            assert isinstance(x, Value)
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
        ins = n.inputs[1:] if n.op in ("gather_rows", "scatter_rows") else n.inputs
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
    return _Tables(ordered, {t for t in ordered if t not in used}, versions)


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
        out = []
        for t in self.tables.params:
            n = t.type.shape[0].folded()
            nm = self.rows_name[t] = names.fresh(f"{t.name}_rows")
            if n.is_static:
                out.append(_Extra(nm, True, _const_param(nm, n.value)))
            elif all(d.buckets for d in n.free_dims()):
                out.append(_Extra(nm, True, _const_param(nm, n)))
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


class _RowLayout(_Layout):
    template = "row"

    def __init__(self, op: KernelOp, tables: _Tables) -> None:
        graph, name = op.graph, op.name
        self.tables = tables
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
        if self.D.is_static:
            d_default: Any = self.D.value
            block_default: Any = next_pow2(self.D.value)
        else:
            d_default = self.D
            block_default = next_pow2(self.D)
        return [
            _Extra(self.D_name, True, _const_param(self.D_name, d_default)),
            _Extra(self.BLOCK_name, True, _const_param(self.BLOCK_name, block_default)),
            *self.table_extras(names),
        ]

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

    def default_num_warps(self) -> int:
        return _row_warps(self.block_max)

    def bucket_num_warps(self) -> "dict[str, int]":
        """One warp count per bucket of ``D``, from the ``BLOCK_D`` of that bucket alone (none for a static ``D``)."""
        if self.D.is_static:
            return {}
        free = sorted(self.D.free_dims(), key=lambda x: x.name)  # the order compile_specs walks the buckets in
        out: dict[str, int] = {}
        for combo in itertools.product(*(x.buckets for x in free)):  # type: ignore[misc]
            value = self.D.subst(dict(zip(free, combo))).value
            out["".join(f".{x.name}{v}" for x, v in zip(free, combo))] = _row_warps(next_pow2(value))
        return out

    def header(self) -> "list[str]":
        d = f"{self.D} (static)" if self.D.is_static else f"{self.D} (bucketed)"
        return [f"template: row; one program per row, grid {_grid_text(self.grid)}", f"D = {d}, BLOCK_D = next_pow2(D)"]

    def load(self, p: str, v: Value) -> str:
        c = self.cls[v]
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

    def __init__(self, op: KernelOp, tables: _Tables) -> None:
        graph, name = op.graph, op.name
        self.tables = tables
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
        self.shape = values[0].type.shape
        for v in values:
            if v.type.shape != self.shape:
                raise _fail(
                    name,
                    "elementwise",
                    "elementwise.same-shape",
                    f"{_label(v)} does not have the shape {_shape_text(self.shape)} of {_label(values[0])}: the "
                    "elementwise template does not broadcast (only python and const scalars combine with a tensor)",
                )
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
        return [
            _Extra(self.n_name, False, Param(self.n_name, 0, "dim", dim=self.n_dim)),
            _Extra(self.BLOCK_name, True, _const_param(self.BLOCK_name, ELEMENTWISE_BLOCK)),
            *self.table_extras(names),
        ]

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
        return [
            f"template: elementwise; every tensor is {_shape_text(self.shape)}, flattened to n = {n} elements",
            f"grid {_grid_text(self.grid)}, BLOCK = {ELEMENTWISE_BLOCK}",
        ]

    def load(self, p: str, v: Value) -> str:
        return f"tl.load({p} + {self.offs}, mask={self.mask}, other=0)"

    def store(self, p: str, v: Value, value: str) -> str:
        return f"tl.store({p} + {self.offs}, {value}, mask={self.mask})"

    def reduce(self, op: str, a: str) -> str:  # pragma: no cover - rejected in __init__
        raise AssertionError("the elementwise template has no reductions")

    def iota(self, v: Value) -> str:
        return self.offs  # the result has the kernel's shape, which is rank 1 (every other iota is broadcast, which this template refuses)


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
        self, op: KernelOp, layout: _Layout, names: _Names, out_names: "list[str]", rename: "dict[str, str]"
    ) -> None:
        self.graph = op.graph
        self.layout = layout
        self.names = names
        self.out_names = out_names
        self.rename = rename  # kernel parameter name -> its name in the generated function, where they differ
        self.params = {p.name: p for p in op.graph.inputs}
        self.var: dict[Value, str] = {}
        self.lines: list[str] = []
        self._count = 0

    def local(self) -> str:
        name = self.names.fresh(f"v{self._count}")
        self._count += 1
        return name

    def emit(self, text: str, comment: str = "") -> None:
        self.lines.append(f"    {text}" + (f"  # {comment}" if comment else ""))

    def operand(self, x: "Value | Scalar") -> str:
        return self.var[x] if isinstance(x, Value) else _scalar_text(x, self.rename)

    def typed(self, x: "Value | Scalar") -> str:
        """An operand of a comparison or a ``where``: a scalar of a dtype that is not Triton's own literal type is a typed constant,
        so that it is rounded to its dtype (Triton would compare a bf16 tensor with an f32 literal, unrounded). A literal is written
        already rounded (through f32, as torch rounds a python float), so the constant is exact whatever Triton makes of it."""
        if not isinstance(x, Scalar) or x.dtype.name in _LITERAL_DTYPES:
            return self.operand(x)
        if x.name is None and x.dtype.is_float:
            x = Scalar(_round_float(float(x.value), x.dtype.name), x.dtype)
        return f"tl.cast({self.operand(x)}, tl.{_TL[x.dtype.name]})"

    def run(self) -> "list[str]":
        graph, layout = self.graph, self.layout
        for line in layout.prologue():
            self.emit(line)
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
        for p in graph.inputs:  # inputs first, in declaration order; an Out parameter is never read, a DevScalar only by tilefish.load
            if p.access != "out" and p in reads and not p.type.devscalar and p not in layout.tables.only:  # a table is read by its gathers
                v = self.local()
                self.var[p] = v
                self.emit(f"{v} = {layout.load(self.rename.get(p.name, p.name), p)}", p.name)
        for n in graph.nodes:
            if n.op == "store":
                self.store(n, last_store)
            elif n.op == "scatter_rows":
                self.scatter(n)
            else:
                (out,) = n.outputs
                v = self.local()
                self.var[out] = v
                rounding = "rounding point" if n.is_rounding_point else ""
                self.emit(f"{v} = {self.expression(n, v)}", rounding)  # an expression may emit the statements it needs first
        for k, v in enumerate(graph.outputs):
            self.emit(layout.store(self.out_names[k], v, self.var[v]), f"return value {k}")
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
        ptr = f"{self.rename.get(t.name, t.name)} + {self.layout.row_offset(t, i)}"
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

    def arg_reduce(self, n: Node, v: str) -> str:
        """``argmax`` / ``argmin`` of an ``[R, D]`` value along the row (the row template; note 8): the index of the first lane that
        holds the extreme value, or of the first NaN when the row has one. Every reduction here is a max or a min, which are exact
        and do not depend on the order of the tree."""
        layout = self.layout
        assert isinstance(layout, _RowLayout)
        mask, cols, block = layout.mask, layout.cols, layout.BLOCK_name
        x = n.inputs[0]
        assert isinstance(x, Value)
        dt = x.type.dtype
        a = self.var[x]
        largest = n.op == "argmax"
        extreme = "tl.max" if largest else "tl.min"
        if dt.name in ("bf16", "f16"):  # widened to f32, exactly
            wide = self.names.fresh(f"{v}_x")
            self.emit(f"{wide} = {a}.to(tl.float32)", "exact")
            a = wide
        best, hit = self.names.fresh(f"{v}_best"), self.names.fresh(f"{v}_hit")
        if dt.is_float:
            nan = self.names.fresh(f"{v}_nan")
            sentinel = 'float("-inf")' if largest else 'float("inf")'
            self.emit(f"{nan} = ({a} != {a}) & {mask}", "a NaN wins")
            self.emit(f"{best} = {extreme}(tl.where({mask}, {a}, {sentinel}), axis=0)")
            self.emit(f"{hit} = tl.where(tl.max({nan}.to(tl.int32), axis=0) > 0, {nan}, ({a} == {best}) & {mask})")
        else:
            lo, hi = dt.int_range()
            self.emit(f"{best} = {extreme}(tl.where({mask}, {a}, {self.typed(Scalar(lo if largest else hi, dt))}), axis=0)")
            self.emit(f"{hit} = ({a} == {best}) & {mask}")
        index = f"tl.min(tl.where({hit}, {cols}, {block}), axis=0)"  # the first lane that holds it
        out = n.outputs[0].type.dtype.name
        return index if out == "i32" else f"{index}.to(tl.{_TL[out]})"

    def expression(self, n: Node, v: str = "") -> str:
        op = n.op
        if op in _ARG_REDUCTIONS:
            return self.arg_reduce(n, v)
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
        a = [self.operand(i) for i in n.inputs]
        if op in _LOGICAL:
            return f"{a[0]} {_LOGICAL[op]} {a[1]}"
        if op == "not":
            return f"~{a[0]}"
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
        if op == "silu":
            return f"{a[0]} * tl.sigmoid({a[0]})"
        if op == "gelu":
            return f"0.5 * {a[0]} * (1.0 + tl.math.erf({a[0]} * 0.7071067811865476))"
        if op == "cast":
            return f"{a[0]}.to(tl.{_TL[n.attrs['to'].name]})"
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
    fn_name = f"tf_{op.name}"
    if not _IDENT.match(fn_name):
        raise _fail(op.name, None, "name", f"'{fn_name}' is not a plain ASCII function name; give the kernel a name of letters, digits and underscores")
    if module is None:
        module = f"{DEFAULT_MODULE}.{op.name}"
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
) -> LoweredKernel:
    """Lower a ``@tilefish.kernel`` to Triton source.

    ``module`` is the Python module the generated source will live in (default
    ``"tilefish_generated.<kernel name>"``); the Triton reference is ``module:tf_<name>``.
    ``template`` forces ``"row"`` or ``"elementwise"``; by default the elementwise template is
    tried first and then the row template. ``num_warps`` overrides the kernel's own
    ``num_warps=`` and the template's rule, and pins the warp count of every bucket; without either,
    each bucket of a bucketed ``D`` gets the warps of its own size. ``fp_fusion`` overrides the kernel's
    own ``fp_fusion=`` (``False``: no FMA contraction). Raises :class:`~tilefish.errors.TilefishLoweringError`
    when the body fits no template.
    """
    module, fn_name = _check_names(op, module)
    if template is not None and template not in TEMPLATES:
        raise TilefishLoweringError(f"lower: template must be 'row' or 'elementwise', got {template!r}", rule="template")
    if fp_fusion is None:
        fp_fusion = op.fp_fusion
    if not isinstance(fp_fusion, bool):
        raise TilefishLoweringError(f"lower: fp_fusion must be True or False, got {fp_fusion!r}", rule="fp-fusion")
    if num_warps is None:
        num_warps = op.num_warps
    if num_warps is not None and (isinstance(num_warps, bool) or not isinstance(num_warps, int) or num_warps not in NUM_WARPS):
        raise TilefishLoweringError(
            f"lower: num_warps must be a power of two from 1 to 32, got {num_warps!r}", rule="num-warps"
        )
    _check_common(op)
    tables = _tables(op)
    errors: list[TilefishLoweringError] = []
    for t in (template,) if template else TEMPLATES:
        try:
            layout = _ElementwiseLayout(op, tables) if t == "elementwise" else _RowLayout(op, tables)
        except TilefishLoweringError as e:
            errors.append(e)
            continue
        return _assemble(op, layout, module, fn_name, num_warps, fp_fusion)
    if len(errors) == 1:
        raise errors[0]
    detail = "; ".join(f"{e.rule}: {str(e).split('failed: ', 1)[-1]}" for e in errors)
    raise TilefishLoweringError(
        f"cannot lower kernel '{op.name}': it fits no template. {detail}", rule="no-template", causes=tuple(errors)
    )


def _assemble(
    op: KernelOp, layout: _Layout, module: str, fn_name: str, num_warps: "int | None", fp_fusion: bool
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
        params.append(_reindexed(p, len(params), new))
        decls.append(f"{new}: tl.constexpr" if p.kind == "const" else new)
        if p.kind == "tensor" and _numel_bounds(p.type.shape)[1] is None:  # type: ignore[union-attr]
            max_elements[new] = INDEX_MAX
    for k, v in enumerate(graph.outputs):
        params.append(Param(out_names[k], len(params), "tensor", "out", type=v.type))
        decls.append(out_names[k])
        if _numel_bounds(v.type.shape)[1] is None:
            max_elements[out_names[k]] = INDEX_MAX
    for e in extras:
        params.append(_reindexed(e.param, len(params), e.param.name))
        decls.append(f"{e.name}: tl.constexpr" if e.constexpr else e.name)
    element_counts: dict[str, str] = {}
    if layout.template == "elementwise":  # n is the element count of every tensor (but a table): say which one a host reads it from
        tables = {rename.get(t.name, t.name) for t in layout.tables.only}
        element_counts[layout.n_name] = next(
            p.name for p in params if p.kind == "tensor" and not p.type.devscalar and p.name not in tables  # type: ignore[union-attr]
        )

    body = _Body(op, layout, names, out_names, rename).run()
    header = [
        f"Generated by tilefish.lower from the kernel '{op.name}'. Do not edit.",
        *layout.header(),
        warps_header,
        *(["fp_fusion: False (no FMA contraction: a * b + c rounds twice)"] if not fp_fusion else []),
        *[f"parameter '{old}' of the kernel is '{new}' here (the name is used by this module)" for old, new in rename.items()],
        "Numerics contract: see the docstring of tilefish.lower.",
    ]
    lines = [f"# {h}" for h in header]
    lines += ["import triton", "import triton.language as tl", "", "", "@triton.jit", f"def {fn_name}("]
    lines += [f"    {d}," for d in decls]
    lines += ["):", *body]
    source = "\n".join(lines) + "\n"

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
    )
    return LoweredKernel(op.name, fn_name, module, layout.template, source, extern, num_warps, op)


def _reindexed(p: Param, index: int, name: str) -> Param:
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
    )
