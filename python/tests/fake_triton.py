"""A pure-Python stand-in for ``triton`` and ``triton.language``, just big enough to execute generated kernels.

It is not Triton: values are Python floats (no f32 rounding between ops, only at ``.to()`` and at
stores), and one program runs at a time. It exists so the tests can check the *logic* of a
generated kernel (masks, reductions over padded lanes, addressing, which value is stored where)
on a machine without a GPU. Out-of-bounds access by an unmasked lane raises ``IndexError``, which is
how a missing mask shows up.
"""

import math
import operator
import struct
import sys
import types


def _f32(x):
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def _f16(x):
    try:
        return struct.unpack("<e", struct.pack("<e", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def _bf16(x):
    if math.isnan(x) or math.isinf(x):
        return x
    bits = struct.unpack("<I", struct.pack("<f", _f32(x)))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16 << 16  # round to nearest even
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


class DType:
    def __init__(self, name, cast):
        self.name, self.cast = name, cast

    def __repr__(self):
        return f"tl.{self.name}"


float32 = DType("float32", _f32)
float16 = DType("float16", _f16)
bfloat16 = DType("bfloat16", _bf16)
int32 = DType("int32", int)
int64 = DType("int64", int)


class constexpr:  # only used as an annotation
    pass


def _div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return math.nan if a == 0 or a != a else math.copysign(math.inf, a) * math.copysign(1.0, b)


class Val:
    """A 0-d value (``scalar``) or a vector of lanes."""

    def __init__(self, data, scalar=False):
        self.data, self.scalar = list(data), scalar

    def lanes(self):
        return len(self.data)

    def _bin(self, other, f, flip=False):
        o = wrap(other)
        a, b = (o, self) if flip else (self, o)
        if a.scalar and b.scalar:
            return Val([f(a.data[0], b.data[0])], True)
        if a.scalar:
            return Val([f(a.data[0], y) for y in b.data])
        if b.scalar:
            return Val([f(x, b.data[0]) for x in a.data])
        assert len(a.data) == len(b.data), "shape mismatch"
        return Val([f(x, y) for x, y in zip(a.data, b.data)])

    def __add__(self, o):
        return self._bin(o, operator.add)

    __radd__ = __add__

    def __sub__(self, o):
        return self._bin(o, operator.sub)

    def __rsub__(self, o):
        return self._bin(o, operator.sub, True)

    def __mul__(self, o):
        return self._bin(o, operator.mul)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self._bin(o, _div)

    def __rtruediv__(self, o):
        return self._bin(o, _div, True)

    def __lt__(self, o):
        return self._bin(o, operator.lt)

    def to(self, dtype):
        return Val([dtype.cast(x) for x in self.data], self.scalar)


def wrap(x):
    if isinstance(x, Val):
        return x
    if isinstance(x, constexpr):
        raise TypeError("constexpr marker used as a value")
    return Val([x], True)


class Buf:
    """A flat buffer with an element dtype; ``buf + offsets`` is a pointer."""

    def __init__(self, data, dtype):
        self.data, self.dtype = [dtype.cast(x) for x in data], dtype

    def __add__(self, offsets):
        return Ptr(self, wrap(offsets))


class Ptr:
    def __init__(self, buf, offs):
        self.buf, self.offs = buf, offs

    def __add__(self, more):
        return Ptr(self.buf, self.offs + more)


def _program_id(axis):
    assert axis == 0
    return Val([_state["pid"]], True)


def _arange(lo, hi):
    return Val(range(lo, hi))


def _load(ptr, mask=None, other=0):
    data = ptr.buf.data
    if ptr.offs.scalar:
        assert mask is None, "a scalar load takes no mask here"
        return Val([data[ptr.offs.data[0]]], True)
    out = []
    for i, off in enumerate(ptr.offs.data):
        if mask is None or wrap(mask).data[i if not wrap(mask).scalar else 0]:
            if not 0 <= off < len(data):
                raise IndexError(f"unmasked load at {off} of a buffer of {len(data)}")
            out.append(data[off])
        else:
            out.append(other)
    return Val(out)


def _store(ptr, value, mask=None):
    buf, v = ptr.buf, wrap(value)
    if ptr.offs.scalar:
        assert mask is None
        assert v.scalar, "a scalar address stores a scalar"
        buf.data[ptr.offs.data[0]] = buf.dtype.cast(v.data[0])
        return
    assert not v.scalar and v.lanes() == ptr.offs.lanes()
    m = wrap(mask) if mask is not None else None
    for i, off in enumerate(ptr.offs.data):
        if m is None or m.data[i]:
            if not 0 <= off < len(buf.data):
                raise IndexError(f"unmasked store at {off} of a buffer of {len(buf.data)}")
            buf.data[off] = buf.dtype.cast(v.data[i])


def _where(mask, a, b):
    m, a, b = wrap(mask), wrap(a), wrap(b)
    n = max(m.lanes(), a.lanes(), b.lanes())
    pick = lambda v, i: v.data[0 if v.scalar else i]  # noqa: E731
    return Val([pick(a, i) if pick(m, i) else pick(b, i) for i in range(n)], m.scalar and a.scalar and b.scalar)


def _reduce(f):
    def reduce(v, axis=0):
        assert axis == 0 and not v.scalar
        return Val([f(v.data)], True)

    return reduce


def _unary(f):
    def apply(v):
        v = wrap(v)
        return Val([f(x) for x in v.data], v.scalar)

    return apply


def _exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


_state = {"pid": 0}

tl = types.ModuleType("triton.language")
tl.constexpr = constexpr
tl.float32, tl.float16, tl.bfloat16, tl.int32, tl.int64 = float32, float16, bfloat16, int32, int64
tl.program_id, tl.arange, tl.load, tl.store, tl.where = _program_id, _arange, _load, _store, _where
tl.sum, tl.max = _reduce(math.fsum), _reduce(max)
tl.exp = _unary(_exp)
tl.rsqrt = _unary(lambda x: _div(1.0, math.sqrt(x)) if x >= 0 else math.nan)
tl.sqrt_rn = _unary(lambda x: math.sqrt(x) if x >= 0 else math.nan)
tl.sigmoid = _unary(lambda x: _div(1.0, 1.0 + _exp(-x)))
tl.math = types.SimpleNamespace(erf=_unary(math.erf))

triton = types.ModuleType("triton")
triton.jit = lambda fn: fn
triton.language = tl


def install(monkeypatch):
    """Make ``import triton`` and ``import triton.language as tl`` find the fakes (undone after the test)."""
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", tl)


def load_kernel(source, fn_name):
    namespace = {}
    exec(compile(source, f"<{fn_name}>", "exec"), namespace)
    return namespace[fn_name]


def launch(kernel, grid, **kwargs):
    for pid in range(grid):
        _state["pid"] = pid
        kernel(**kwargs)
