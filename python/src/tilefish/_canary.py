"""The kernels of a cuDNN convolution's canary (``docs/en/manifest-schema-3.md``, section 9.4): the operands, made on the device, and the digest of ``y``.

A canary is ``{"seed": s, "y_digest": d}``: the pin run on operands made of the seed at the entry's shape gives a ``y`` whose digest is ``d``. Both
are made on the device by the PTX below (:data:`CANARY_PTX`, the same text as the crate's ``src/canary.ptx``: a test holds them equal), which the
driver compiles for the device it is loaded on; :func:`canary_bytes` and :func:`digest_of` are the same computations on the host (a reference, for
the tests and the vectors the Rust host is held to).

* ``mix(v)``: splitmix64's finalizer of ``v + 0x9E3779B97F4A7C15`` (modulo 2**64).
* Element ``i`` of an operand (tag 0 for ``x``, 1 for ``w``): ``h = mix(base + i)``, ``base = mix(2 * seed + tag)``; its low 16 bits (f16, bf16) or
  32 bits (f32), with the most significant byte ``b`` of the element mapped: f16 ``(b & 0x80) | ((7 + ((b >> 2) & 7)) << 2) | (b & 3)`` (magnitudes
  in [2**-8, 1)), bf16 and f32 ``(b & 0x80) | (55 + (b & 7))`` (magnitudes in [2**-17, 2**-1)); little-endian, dense in the layout.
* The digest of ``y``: its bytes as 32-bit little-endian words ``w_j`` (zero-padded to a whole word), ``s1 = sum w_j * (mix(j ^ K1) | 1)`` and
  ``s2 = sum w_j * (mix(j ^ K2) | 1)`` modulo 2**64, ``K1 = 0x243F6A8885A308D3``, ``K2 = 0x13198A2E03707344``; ``d`` is ``s1`` then ``s2`` in 16
  lowercase hex digits each. A sum does not depend on its order, so the device adds it in any grid; a change of one word always changes it (the
  multipliers are odd), and of several words with a chance of about 2**-128.
"""

from __future__ import annotations

import ctypes
import struct
import threading
from typing import Any

M64 = (1 << 64) - 1
GOLDEN, MUL1, MUL2 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
K1, K2 = 0x243F6A8885A308D3, 0x13198A2E03707344
KINDS = {"f16": 0, "bf16": 1, "f32": 2}
ELEMENT_BYTES = {"f16": 2, "bf16": 2, "f32": 4}
BLOCK = 256
MAX_FILL_BLOCKS = 65535
MAX_DIGEST_BLOCKS = 1024


def mix(v: int) -> int:
    z = (v + GOLDEN) & M64
    z = ((z ^ (z >> 30)) * MUL1) & M64
    z = ((z ^ (z >> 27)) * MUL2) & M64
    return z ^ (z >> 31)


def base(seed: int, tag: int) -> int:
    """The base of an operand's elements: ``mix(2 * seed + tag)``."""
    return mix(2 * seed + tag)


def _top16(dtype: str, b: int) -> int:
    if dtype == "f16":
        return (b & 0x80) | ((7 + ((b >> 2) & 7)) << 2) | (b & 3)
    return (b & 0x80) | (55 + (b & 7))


def canary_bytes(dtype: str, seed: int, tag: int, elements: int) -> bytes:
    """The bytes of the first ``elements`` elements of a canary operand, on the host (what the fill kernel writes)."""
    first = base(seed, tag)
    if dtype == "f32":
        out = []
        for i in range(elements):
            r = mix((first + i) & M64) & 0xFFFFFFFF
            out.append((_top16(dtype, r >> 24) << 24) | (r & 0xFFFFFF))
        return struct.pack(f"<{elements}I", *out)
    halves = []
    for i in range(elements):
        r = mix((first + i) & M64) & 0xFFFF
        halves.append((_top16(dtype, r >> 8) << 8) | (r & 0xFF))
    return struct.pack(f"<{elements}H", *halves)


def digest_of(data: bytes) -> str:
    """The digest of ``y``'s bytes, on the host (what the digest kernel adds)."""
    padded = data + b"\x00" * (-len(data) % 4)
    words = struct.unpack(f"<{len(padded) // 4}I", padded)
    s1 = s2 = 0
    for j, w in enumerate(words):
        s1 = (s1 + w * (mix(j ^ K1) | 1)) & M64
        s2 = (s2 + w * (mix(j ^ K2) | 1)) & M64
    return f"{s1:016x}{s2:016x}"


# -- on the device ----------------------------------------------------------------------------------------------------------------------------


_lock = threading.Lock()
_functions: "dict[int, tuple[Any, Any, Any]]" = {}


def _kernels(device_index: int) -> "tuple[Any, Any]":
    """The fill and digest kernels in the current context of device ``device_index`` (loaded once per device: the driver compiles the PTX)."""
    from ._cuda import _check, _driver

    with _lock:
        found = _functions.get(device_index)
        if found is None:
            drv = _driver()
            (module,) = _check(drv.cuModuleLoadData(CANARY_PTX.encode("ascii") + b"\x00"), "cuModuleLoadData(canary PTX)")
            (fill,) = _check(drv.cuModuleGetFunction(module, b"tilefish_canary_fill"), "cuModuleGetFunction(tilefish_canary_fill)")
            (digest,) = _check(drv.cuModuleGetFunction(module, b"tilefish_canary_digest"), "cuModuleGetFunction(tilefish_canary_digest)")
            found = _functions[device_index] = (module, fill, digest)
    return found[1], found[2]


def _launch(function: Any, blocks: int, params: "list[Any]", stream: int, what: str) -> None:
    from ._cuda import _check, _driver

    drv = _driver()
    slots = (ctypes.c_void_p * len(params))(*[ctypes.addressof(p) for p in params])
    _check(drv.cuLaunchKernel(function, blocks, 1, 1, BLOCK, 1, 1, 0, drv.CUstream(stream), ctypes.addressof(slots), 0), what)


def fill(address: int, elements: int, dtype: str, seed: int, tag: int, stream: int, device_index: int) -> None:
    """Write the canary operand of ``seed`` and ``tag`` (``elements`` elements of ``dtype``) at ``address``, enqueued on ``stream``."""
    kernel, _ = _kernels(device_index)
    blocks = max(1, min(-(-elements // BLOCK), MAX_FILL_BLOCKS))
    params = [ctypes.c_uint64(address), ctypes.c_uint64(elements), ctypes.c_uint64(base(seed, tag)), ctypes.c_uint32(KINDS[dtype])]
    _launch(kernel, blocks, params, stream, "cuLaunchKernel(tilefish_canary_fill)")


def digest(address: int, words: int, out: int, stream: int, device_index: int) -> None:
    """Add the digest of the ``words`` 32-bit words at ``address`` to the two u64 at ``out`` (zeroed by the caller), enqueued on ``stream``."""
    _, kernel = _kernels(device_index)
    blocks = max(1, min(-(-words // BLOCK), MAX_DIGEST_BLOCKS))
    params = [ctypes.c_uint64(address), ctypes.c_uint64(words), ctypes.c_uint64(out)]
    _launch(kernel, blocks, params, stream, "cuLaunchKernel(tilefish_canary_digest)")


def hex_of(s1: int, s2: int) -> str:
    """The digest's text of the two sums."""
    return f"{s1 & M64:016x}{s2 & M64:016x}"

# the crate's src/canary.ptx, character for character (test_library_conv.py holds them equal)
CANARY_PTX = """// The kernels of a cuDNN convolution's canary (docs/en/manifest-schema-3.md, section 9.4): the operands, made on the device, and the digest of y.
// The same text is in the Python package (tilefish/_canary.py, CANARY_PTX) and in the crate (src/canary.ptx, include_str!): a test holds them
// equal. The driver compiles it for the device it is loaded on (PTX for sm_70 and later), so neither host needs a compiler.
//
// mix(v) is splitmix64's finalizer of v + 0x9E3779B97F4A7C15 (constants written as nvcc writes 64-bit constants: signed decimal).
//
// tilefish_canary_fill(out, count, base, kind): element i (i < count) of an operand is made of h = mix(base + i) (wrapping): its low 16 bits for
// f16 (kind 0) and bf16 (kind 1), its low 32 bits for f32 (kind 2), the most significant byte b of the element mapped: f16
// (b & 0x80) | ((7 + ((b >> 2) & 7)) << 2) | (b & 3), bf16 and f32 (b & 0x80) | (55 + (b & 7)). Stored little-endian at out + i * size.
//
// tilefish_canary_digest(words, count, out): over the count 32-bit little-endian words w_j of y (zero-padded to a whole word), two sums modulo
// 2^64, s1 = sum w_j * (mix(j ^ K1) | 1), s2 = sum w_j * (mix(j ^ K2) | 1), K1 = 0x243F6A8885A308D3, K2 = 0x13198A2E03707344, added with
// atomics to out[0] and out[1] (which start at zero): an order-independent sum, so the digest is the same for any grid.

.version 7.0
.target sm_70
.address_size 64

.visible .entry tilefish_canary_fill(
	.param .u64 tilefish_canary_fill_out,
	.param .u64 tilefish_canary_fill_count,
	.param .u64 tilefish_canary_fill_base,
	.param .u32 tilefish_canary_fill_kind
)
{
	.reg .pred 	%p<4>;
	.reg .b32 	%r<12>;
	.reg .b64 	%rd<12>;

	ld.param.u64 	%rd1, [tilefish_canary_fill_out];
	ld.param.u64 	%rd2, [tilefish_canary_fill_count];
	ld.param.u64 	%rd3, [tilefish_canary_fill_base];
	ld.param.u32 	%r1, [tilefish_canary_fill_kind];
	cvta.to.global.u64 	%rd1, %rd1;
	mov.u32 	%r2, %ctaid.x;
	mov.u32 	%r3, %ntid.x;
	mov.u32 	%r4, %tid.x;
	mov.u32 	%r5, %nctaid.x;
	mul.wide.u32 	%rd4, %r2, %r3;
	cvt.u64.u32 	%rd5, %r4;
	add.u64 	%rd4, %rd4, %rd5;
	mul.wide.u32 	%rd6, %r5, %r3;
$FILL_LOOP:
	setp.ge.u64 	%p1, %rd4, %rd2;
	@%p1 bra 	$FILL_DONE;
	add.u64 	%rd7, %rd3, %rd4;
	add.u64 	%rd8, %rd7, -7046029254386353131;
	shr.u64 	%rd9, %rd8, 30;
	xor.b64 	%rd8, %rd8, %rd9;
	mul.lo.u64 	%rd8, %rd8, -4658895280553007687;
	shr.u64 	%rd9, %rd8, 27;
	xor.b64 	%rd8, %rd8, %rd9;
	mul.lo.u64 	%rd8, %rd8, -7723592293110705685;
	shr.u64 	%rd9, %rd8, 31;
	xor.b64 	%rd8, %rd8, %rd9;
	cvt.u32.u64 	%r6, %rd8;
	setp.eq.u32 	%p2, %r1, 2;
	@%p2 bra 	$FILL_F32;
	and.b32 	%r7, %r6, 255;
	shr.u32 	%r8, %r6, 8;
	and.b32 	%r8, %r8, 255;
	and.b32 	%r9, %r8, 128;
	setp.eq.u32 	%p3, %r1, 0;
	@%p3 bra 	$FILL_F16;
	and.b32 	%r10, %r8, 7;
	add.u32 	%r10, %r10, 55;
	or.b32 	%r10, %r10, %r9;
	bra.uni 	$FILL_STORE16;
$FILL_F16:
	shr.u32 	%r10, %r8, 2;
	and.b32 	%r10, %r10, 7;
	add.u32 	%r10, %r10, 7;
	shl.b32 	%r10, %r10, 2;
	or.b32 	%r10, %r10, %r9;
	and.b32 	%r11, %r8, 3;
	or.b32 	%r10, %r10, %r11;
$FILL_STORE16:
	shl.b32 	%r10, %r10, 8;
	or.b32 	%r10, %r10, %r7;
	shl.b64 	%rd10, %rd4, 1;
	add.u64 	%rd10, %rd1, %rd10;
	st.global.u16 	[%rd10], %r10;
	bra.uni 	$FILL_NEXT;
$FILL_F32:
	shr.u32 	%r8, %r6, 24;
	and.b32 	%r9, %r8, 128;
	and.b32 	%r10, %r8, 7;
	add.u32 	%r10, %r10, 55;
	or.b32 	%r10, %r10, %r9;
	shl.b32 	%r10, %r10, 24;
	and.b32 	%r11, %r6, 16777215;
	or.b32 	%r10, %r10, %r11;
	shl.b64 	%rd10, %rd4, 2;
	add.u64 	%rd10, %rd1, %rd10;
	st.global.u32 	[%rd10], %r10;
$FILL_NEXT:
	add.u64 	%rd4, %rd4, %rd6;
	bra.uni 	$FILL_LOOP;
$FILL_DONE:
	ret;
}

.visible .entry tilefish_canary_digest(
	.param .u64 tilefish_canary_digest_words,
	.param .u64 tilefish_canary_digest_count,
	.param .u64 tilefish_canary_digest_out
)
{
	.reg .pred 	%p<4>;
	.reg .b32 	%r<12>;
	.reg .b64 	%rd<20>;

	ld.param.u64 	%rd1, [tilefish_canary_digest_words];
	ld.param.u64 	%rd2, [tilefish_canary_digest_count];
	ld.param.u64 	%rd3, [tilefish_canary_digest_out];
	cvta.to.global.u64 	%rd1, %rd1;
	cvta.to.global.u64 	%rd3, %rd3;
	mov.u32 	%r2, %ctaid.x;
	mov.u32 	%r3, %ntid.x;
	mov.u32 	%r4, %tid.x;
	mov.u32 	%r5, %nctaid.x;
	mul.wide.u32 	%rd4, %r2, %r3;
	cvt.u64.u32 	%rd5, %r4;
	add.u64 	%rd4, %rd4, %rd5;
	mul.wide.u32 	%rd6, %r5, %r3;
	mov.u64 	%rd11, 0;
	mov.u64 	%rd12, 0;
$DIGEST_LOOP:
	setp.ge.u64 	%p1, %rd4, %rd2;
	@%p1 bra 	$DIGEST_REDUCE;
	shl.b64 	%rd7, %rd4, 2;
	add.u64 	%rd7, %rd1, %rd7;
	ld.global.u32 	%r6, [%rd7];
	cvt.u64.u32 	%rd13, %r6;
	xor.b64 	%rd8, %rd4, 2611923443488327891;
	add.u64 	%rd8, %rd8, -7046029254386353131;
	shr.u64 	%rd9, %rd8, 30;
	xor.b64 	%rd8, %rd8, %rd9;
	mul.lo.u64 	%rd8, %rd8, -4658895280553007687;
	shr.u64 	%rd9, %rd8, 27;
	xor.b64 	%rd8, %rd8, %rd9;
	mul.lo.u64 	%rd8, %rd8, -7723592293110705685;
	shr.u64 	%rd9, %rd8, 31;
	xor.b64 	%rd8, %rd8, %rd9;
	or.b64 	%rd8, %rd8, 1;
	mul.lo.u64 	%rd8, %rd8, %rd13;
	add.u64 	%rd11, %rd11, %rd8;
	xor.b64 	%rd8, %rd4, 1376283091369227076;
	add.u64 	%rd8, %rd8, -7046029254386353131;
	shr.u64 	%rd9, %rd8, 30;
	xor.b64 	%rd8, %rd8, %rd9;
	mul.lo.u64 	%rd8, %rd8, -4658895280553007687;
	shr.u64 	%rd9, %rd8, 27;
	xor.b64 	%rd8, %rd8, %rd9;
	mul.lo.u64 	%rd8, %rd8, -7723592293110705685;
	shr.u64 	%rd9, %rd8, 31;
	xor.b64 	%rd8, %rd8, %rd9;
	or.b64 	%rd8, %rd8, 1;
	mul.lo.u64 	%rd8, %rd8, %rd13;
	add.u64 	%rd12, %rd12, %rd8;
	add.u64 	%rd4, %rd4, %rd6;
	bra.uni 	$DIGEST_LOOP;
$DIGEST_REDUCE:
	mov.b64 	{%r7, %r8}, %rd11;
	shfl.sync.down.b32 	%r9, %r7, 16, 31, -1;
	shfl.sync.down.b32 	%r10, %r8, 16, 31, -1;
	mov.b64 	%rd14, {%r9, %r10};
	add.u64 	%rd11, %rd11, %rd14;
	mov.b64 	{%r7, %r8}, %rd11;
	shfl.sync.down.b32 	%r9, %r7, 8, 31, -1;
	shfl.sync.down.b32 	%r10, %r8, 8, 31, -1;
	mov.b64 	%rd14, {%r9, %r10};
	add.u64 	%rd11, %rd11, %rd14;
	mov.b64 	{%r7, %r8}, %rd11;
	shfl.sync.down.b32 	%r9, %r7, 4, 31, -1;
	shfl.sync.down.b32 	%r10, %r8, 4, 31, -1;
	mov.b64 	%rd14, {%r9, %r10};
	add.u64 	%rd11, %rd11, %rd14;
	mov.b64 	{%r7, %r8}, %rd11;
	shfl.sync.down.b32 	%r9, %r7, 2, 31, -1;
	shfl.sync.down.b32 	%r10, %r8, 2, 31, -1;
	mov.b64 	%rd14, {%r9, %r10};
	add.u64 	%rd11, %rd11, %rd14;
	mov.b64 	{%r7, %r8}, %rd11;
	shfl.sync.down.b32 	%r9, %r7, 1, 31, -1;
	shfl.sync.down.b32 	%r10, %r8, 1, 31, -1;
	mov.b64 	%rd14, {%r9, %r10};
	add.u64 	%rd11, %rd11, %rd14;
	mov.b64 	{%r7, %r8}, %rd12;
	shfl.sync.down.b32 	%r9, %r7, 16, 31, -1;
	shfl.sync.down.b32 	%r10, %r8, 16, 31, -1;
	mov.b64 	%rd14, {%r9, %r10};
	add.u64 	%rd12, %rd12, %rd14;
	mov.b64 	{%r7, %r8}, %rd12;
	shfl.sync.down.b32 	%r9, %r7, 8, 31, -1;
	shfl.sync.down.b32 	%r10, %r8, 8, 31, -1;
	mov.b64 	%rd14, {%r9, %r10};
	add.u64 	%rd12, %rd12, %rd14;
	mov.b64 	{%r7, %r8}, %rd12;
	shfl.sync.down.b32 	%r9, %r7, 4, 31, -1;
	shfl.sync.down.b32 	%r10, %r8, 4, 31, -1;
	mov.b64 	%rd14, {%r9, %r10};
	add.u64 	%rd12, %rd12, %rd14;
	mov.b64 	{%r7, %r8}, %rd12;
	shfl.sync.down.b32 	%r9, %r7, 2, 31, -1;
	shfl.sync.down.b32 	%r10, %r8, 2, 31, -1;
	mov.b64 	%rd14, {%r9, %r10};
	add.u64 	%rd12, %rd12, %rd14;
	mov.b64 	{%r7, %r8}, %rd12;
	shfl.sync.down.b32 	%r9, %r7, 1, 31, -1;
	shfl.sync.down.b32 	%r10, %r8, 1, 31, -1;
	mov.b64 	%rd14, {%r9, %r10};
	add.u64 	%rd12, %rd12, %rd14;
	and.b32 	%r11, %r4, 31;
	setp.ne.u32 	%p2, %r11, 0;
	@%p2 bra 	$DIGEST_DONE;
	atom.global.add.u64 	%rd15, [%rd3], %rd11;
	atom.global.add.u64 	%rd15, [%rd3+8], %rd12;
$DIGEST_DONE:
	ret;
}
"""
