"""Cast kernels and the values they must give (not collected: the name has no ``test_`` prefix).

The tables are what the binaries do on an sm_103 GPU with Triton 3.5.1, measured by ``test_gpu_numerics.py``,
which asserts them on the GPU; ``test_reference.py`` asserts that the torch interpreter gives the same on the
CPU. No ``from __future__ import annotations`` here: tilefish evaluates annotations at declaration.
"""

from lower_cases import T

import tilefish
from tilefish import Tensor, bf16, e4m3, e5m2, f16, f32, i32, i64, u8


@tilefish.kernel
def to_u8(x: Tensor[T, f32]) -> Tensor[T, u8]:
    return x.to(u8)


@tilefish.kernel
def to_i32(x: Tensor[T, f32]) -> Tensor[T, i32]:
    return x.to(i32)


@tilefish.kernel
def to_i64(x: Tensor[T, f32]) -> Tensor[T, i64]:
    return x.to(i64)


@tilefish.kernel
def to_e4m3(x: Tensor[T, f32]) -> Tensor[T, e4m3]:
    return x.to(e4m3)


@tilefish.kernel
def to_e5m2(x: Tensor[T, f32]) -> Tensor[T, e5m2]:
    return x.to(e5m2)


@tilefish.kernel
def to_f16(x: Tensor[T, f32]) -> Tensor[T, f16]:
    return x.to(f16)


@tilefish.kernel
def to_bf16(x: Tensor[T, f32]) -> Tensor[T, bf16]:
    return x.to(bf16)


NAN, INF = float("nan"), float("inf")

# f32 -> u8: truncate toward zero, clamp to [0, 65535] (negative and NaN give 0), keep the low 8 bits
U8_INPUT = [-1000.0, -256.0, -129.0, -128.5, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 0.9999, 1.0, 1.5, 1.7, 127.5, 128.0,
            254.5, 254.9, 255.0, 255.5, 256.0, 257.0, 300.0, 1000.0, 65536.0, 1e10, 3e9, -3e9, INF, -INF, NAN, 2.5]
U8_OUTPUT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 127, 128, 254, 254, 255, 255, 0, 1, 44, 232, 255, 255, 255, 0, 255, 0, 0, 2]

# f32 -> i32: truncate toward zero, saturate at both ends, NaN is 0
I32_INPUT = [-1e10, -3e9, -2147483904.0, -2147483648.0, -1.9, -0.5, 0.0, 0.5, 1.9, 2147483520.0, 2147483648.0, 3e9, 1e19, INF, -INF, NAN]
I32_OUTPUT = [-(2**31), -(2**31), -(2**31), -(2**31), -1, 0, 0, 0, 1, 2147483520, 2**31 - 1, 2**31 - 1, 2**31 - 1, 2**31 - 1, -(2**31), 0]

# f32 -> i64: the same, except that NaN is the most negative value
I64_INPUT = [-1e19, -3e9, -1.9, 0.5, 1.9, 3e9, 1e19, 1e30, INF, -INF, NAN]
I64_OUTPUT = [-(2**63), -3000000000, -1, 0, 1, 3000000000, 2**63 - 1, 2**63 - 1, 2**63 - 1, -(2**63), -(2**63)]

# f32 -> e4m3 (the bytes): round to nearest even, saturate at +-448 (never NaN or infinity for a finite or infinite input)
E4M3_TABLE = [(447.0, 0x7E), (448.0, 0x7E), (449.0, 0x7E), (465.0, 0x7E), (1000.0, 0x7E), (1e5, 0x7E), (-448.0, 0xFE),
              (-500.0, 0xFE), (INF, 0x7E), (-INF, 0xFE), (NAN, 0x7F), (1.0, 0x38), (0.001, 0x01), (1e-5, 0x00)]

# f32 -> e5m2 (the bytes): round to nearest even, saturate at +-57344
E5M2_TABLE = [(448.0, 0x5F), (480.0, 0x60), (1000.0, 0x64), (57344.0, 0x7B), (61440.0, 0x7B), (1e5, 0x7B), (-1e5, 0xFB),
              (-500.0, 0xE0), (INF, 0x7B), (-INF, 0xFB), (NAN, 0x7F), (1.0, 0x3C), (1e-5, 0x01), (1e-9, 0x00)]

# values for the casts that are plain round to nearest even, overflow to infinity (f16 at 65520, bf16 near 3.4e38)
HALF_INPUT = [-1e10, -1.9, -0.5, 0.0, 0.5, 1.9, 65504.0, 65519.0, 65520.0, 70000.0, 3.4e38, 3.3895e38, 3.39e38, -3.4e38, 1e-8, 1e-5, INF, -INF, NAN]
