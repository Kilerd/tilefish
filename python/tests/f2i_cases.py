"""Float to integer casts with an explicit rounding and saturation (note 15 of ``tilefish.lower``), shared by the lowering tests and the
gpu tests (not collected), and an exact model of what each mode gives, in plain Python (no torch).

No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

import math
import struct

import tilefish
from tilefish import Dim, Tensor, bf16, e4m3, e5m2, f16, f32, i32, i64, u8

N = Dim("N")
R = Dim("R")
D = Dim("D", 256)

ROUNDINGS = ("rn", "rz", "rd", "ru")
INTS = {"i32": i32, "i64": i64, "u8": u8}


def _all_roundings(name, dtype, saturate):
    """A kernel of an f32 tensor that returns its four casts to ``dtype`` (``rn``, ``rz``, ``rd``, ``ru``) in one saturation mode."""

    def body(x: Tensor[N, f32]) -> tuple[Tensor[N, dtype], Tensor[N, dtype], Tensor[N, dtype], Tensor[N, dtype]]:
        return tuple(x.to(dtype, rounding=r, saturate=saturate) for r in ROUNDINGS)

    return tilefish.kernel(body, name=name)


i32_sat = _all_roundings("i32_sat", i32, True)
i32_wrap = _all_roundings("i32_wrap", i32, False)
i64_sat = _all_roundings("i64_sat", i64, True)
i64_wrap = _all_roundings("i64_wrap", i64, False)
u8_sat = _all_roundings("u8_sat", u8, True)
u8_wrap = _all_roundings("u8_wrap", u8, False)
BY_MODE = {("i32", True): i32_sat, ("i32", False): i32_wrap, ("i64", True): i64_sat, ("i64", False): i64_wrap, ("u8", True): u8_sat, ("u8", False): u8_wrap}

# every (integer type, rounding, saturate) of a cast from a narrower float: i64 has one mode, so it appears once per rounding
MODES = [(dt, r, s) for dt in ("i32", "u8") for s in (True, False) for r in ROUNDINGS] + [("i64", r, True) for r in ROUNDINGS]


def _from(name, src):
    """A kernel of a ``src`` tensor (bf16, f16, e4m3, e5m2) that returns its casts in every mode of :data:`MODES`: the source is widened to f32
    first, exactly."""
    outs = tuple(Tensor[N, INTS[dt]] for dt, _, _ in MODES)

    def body(x: Tensor[N, src]) -> tuple[outs]:
        return tuple(x.to(INTS[dt], rounding=r, saturate=s) for dt, r, s in MODES)

    return tilefish.kernel(body, name=name)


from_bf16 = _from("from_bf16", bf16)
from_f16 = _from("from_f16", f16)
from_e4m3 = _from("from_e4m3", e4m3)
from_e5m2 = _from("from_e5m2", e5m2)


@tilefish.kernel
def quantize_rows(x: Tensor[R, D, f32], scale: Tensor[R, 1, f32]) -> tuple[Tensor[R, D, u8], Tensor[R, D, i32]]:
    """The row template: a scaled row rounded to the nearest u8 (saturating), and its distance from the row's maximum rounded down,
    wrapping."""
    q = (x * scale + 128.0).to(u8, rounding="rn")
    return q, (x - tilefish.max(x, axis=-1, keepdims=True)).to(i32, rounding="rd", saturate=False)


# -- the model ------------------------------------------------------------------------------------------------------------------------


def f32_of(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def bits_of(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


_RANGE = {"i32": (-(2**31), 2**31 - 1), "i64": (-(2**63), 2**63 - 1), "u8": (0, 255)}
_BITS = {"i32": 32, "i64": 64, "u8": 8}


def expected(value, dtype, rounding, saturate):
    """What the cast of the float ``value`` gives (note 15), exactly: round to an integer (Python's ``round`` of a float is to nearest even),
    a NaN is 0 and the result is clamped to the ``i64`` range (an infinity to its bound); ``saturate`` clamps it to ``dtype``'s range, else
    its low bits are kept (two's complement)."""
    lo64, hi64 = _RANGE["i64"]
    if math.isnan(value):
        r = 0
    elif math.isinf(value):
        r = hi64 if value > 0 else lo64
    else:
        r = {"rn": round, "rz": math.trunc, "rd": math.floor, "ru": math.ceil}[rounding](value)
        r = max(lo64, min(hi64, r))
    lo, hi = _RANGE[dtype]
    if saturate or dtype == "i64":
        return max(lo, min(hi, r))
    w = r % (1 << _BITS[dtype])
    return w - (1 << _BITS[dtype]) if lo < 0 and w > hi else w


def _f32(v):
    return f32_of(bits_of(v))


# adversarial f32 inputs, as bit patterns: ties, the bounds of every type and their neighbours, subnormals, infinities, NaNs with payloads
ADVERSARIAL = [bits_of(_f32(v)) for v in (
    0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5, 0.49999997, -0.49999997, 0.50000006, -0.50000006, 1.0000001, -1.0000001,
    1e-45, -1e-45, 1.1754942e-38, -1.1754942e-38, 1.1754944e-38, -1.1754944e-38, 1e-30, -1e-30,
    -1.0, -128.5, -129.5, 127.5, 128.5, 254.5, 254.50002, 255.0, 255.49998, 255.5, 256.0, 256.5, 300.0, 1000.0, 65535.5, 65536.0,
    8388607.5, -8388607.5, 8388608.0, 16777215.0, 16777216.0,
    2147483520.0, -2147483520.0, 2147483648.0, -2147483648.0, -2147483904.0, 4294967296.0, -4294967296.0, 4294967552.0,
    9.223371487098962e18, -9.223371487098962e18, 9.223372036854776e18, -9.223372036854776e18, 1.8446744073709552e19, -1.8446744073709552e19,
    1e30, -1e30, 3.4028234663852886e38, -3.4028234663852886e38, math.inf, -math.inf,
)] + [0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF, 0x7FC00001]
