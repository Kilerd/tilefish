"""The cuBLASLt binding (``tilefish._cublaslt``) without a GPU, without torch and without cuBLASLt.

What can be checked on any machine: the hex form of a pinned algorithm (round trip and strictness), the cuBLAS-level problem
that tilefish's ``D[M, N] = A[M, K] @ B[N, K]^T`` is posed as (``describe``, checked by running a column-major BLAS gemm
written in plain Python on the values it names), the exact descriptors and call arguments the binding builds (against a
recording stand-in for the library), the validation of every argument before the library is touched, how the library is found
(mapped, loaded by torch, in the nvidia-cublas wheel, or missing) and what an error from the library looks like. The same
operation on a real GPU is ``test_gpu_cublaslt.py``.
"""

import contextlib
import ctypes
import os
import subprocess
import sys
import types
from unittest import mock

import pytest

from tilefish import TilefishRuntimeError
from tilefish import _cublaslt as L

HEX = "".join(f"{i:02x}" for i in range(64))


@pytest.fixture(autouse=True)
def _no_library_is_loaded(monkeypatch):
    """Every test starts without an opened library, also when a GPU test of the same session has opened the real one."""
    monkeypatch.setattr(L, "_library", None)


# -- the hex form ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [bytes(64), b"\xff" * 64, bytes(range(64)), bytes(range(255, 191, -1)), (b"\x01\x80" * 32)])
def test_hex_round_trip(raw):
    text = L.algo_to_hex(raw)
    assert len(text) == 128 and text == text.lower()
    assert L.algo_from_hex(text) == raw
    assert L.algo_to_hex(L.algo_from_hex(text)) == text


def test_hex_is_the_bytes_in_memory_order():
    assert L.algo_to_hex(bytes(range(64))) == HEX
    assert L.algo_to_hex(b"\x42" + bytes(63)).startswith("42000000")  # the first uint64 is little-endian, as in memory


def test_hex_accepts_a_bytearray_and_returns_bytes():
    assert L.algo_to_hex(bytearray(range(64))) == HEX
    assert type(L.algo_from_hex(HEX)) is bytes


@pytest.mark.parametrize("size", [0, 1, 63, 65, 128])
def test_algo_to_hex_is_strict_about_the_length(size):
    with pytest.raises(TilefishRuntimeError, match="exactly 64 bytes"):
        L.algo_to_hex(bytes(size))


@pytest.mark.parametrize("value", [HEX, [0] * 64, None, 5, memoryview(bytes(64))])
def test_algo_to_hex_is_strict_about_the_type(value):
    with pytest.raises(TilefishRuntimeError):
        L.algo_to_hex(value)


@pytest.mark.parametrize(
    "text",
    [
        "",
        HEX[:-1],
        HEX + "0",
        HEX.upper(),
        HEX[:-1] + "A",
        HEX[:-1] + "g",
        "0x" + HEX[2:],
        " " + HEX[1:],
        HEX + "\n",
        HEX[:-1] + "\n",
        HEX[:60] + " " * 8 + HEX[68:],
        "٠" * 128,  # Arabic-Indic digits are digits to str.isdigit() and not hex
        HEX[:-1] + "é",
        "-" + HEX[1:],
    ],
)
def test_algo_from_hex_is_strict(text):
    with pytest.raises(TilefishRuntimeError, match="128 lowercase hex characters"):
        L.algo_from_hex(text)


@pytest.mark.parametrize("value", [None, 5, bytes.fromhex(HEX), HEX.encode(), [HEX]])
def test_algo_from_hex_wants_a_str(value):
    with pytest.raises(TilefishRuntimeError, match="string"):
        L.algo_from_hex(value)


# -- describe: the translation to column-major, as numbers ---------------------------------------------------------------------


def test_describe_for_a_weight_shaped_problem():
    d = L.describe(64, 3072, 1024, False)
    assert d["transa"] == "T" and d["transb"] == "N"
    assert d["compute_type"] == "CUBLAS_COMPUTE_32F" and d["scale_type"] == "CUDA_R_32F"
    assert (d["cublas_m"], d["cublas_n"], d["cublas_k"]) == (3072, 64, 1024)  # m and n swap: row-major D is column-major D^T
    assert (d["lda"], d["ldb"], d["ldc"], d["ldd"]) == (1024, 1024, 3072, 3072)
    assert d["epilogue"] == "none" and d["bias_type"] is None and d["bias_length"] is None
    assert d["dtype"] == "bf16"
    assert d["alpha"] == 1.0 and d["beta"] == 0.0 and d["pointer_mode"] == "host" and d["order"] == "col"


def test_describe_with_a_bias():
    d = L.describe(1, 8, 16, True)
    assert (d["cublas_m"], d["cublas_n"], d["cublas_k"]) == (8, 1, 16)
    assert (d["lda"], d["ldb"], d["ldc"], d["ldd"]) == (16, 16, 8, 8)
    assert d["epilogue"] == "bias" and d["bias_type"] == "CUDA_R_16BF"
    assert d["bias_length"] == 8  # one value per row of the column-major D: bias[N]


def test_describe_names_the_operands_and_the_types():
    d = L.describe(2, 3, 4, False)
    assert d["a_operand"] == "B[N, K]" and d["b_operand"] == "A[M, K]"
    assert {d["a_type"], d["b_type"], d["c_type"], d["d_type"]} == {"CUDA_R_16BF"}
    assert d["reduction_schemes"] == [0, 2]  # no split-K, or split-K with f32 partial sums


def test_describe_changes_only_with_the_problem():
    assert L.describe(3, 4, 5, False) == L.describe(3, 4, 5, False)
    assert L.describe(3, 4, 5, False) != L.describe(3, 4, 5, True)
    assert L.describe(3, 4, 5, False) != L.describe(4, 3, 5, False)


@pytest.mark.parametrize("m,n,k", [(1, 1, 1), (3, 4, 5), (5, 2, 3), (7, 1, 2), (2, 9, 1), (6, 6, 6)])
@pytest.mark.parametrize("bias", [False, True])
def test_describe_leading_dimensions_are_legal_blas_ones(m, n, k, bias):
    d = L.describe(m, n, k, bias)
    # a TN gemm: A is stored k x cublas_m (transa = T), B is stored k x cublas_n (transb = N), D is cublas_m x cublas_n
    assert d["lda"] >= d["cublas_k"] and d["ldb"] >= d["cublas_k"]
    assert d["ldd"] >= d["cublas_m"] and d["ldc"] == d["ldd"]


def _col_major_gemm(*, transa, transb, m, n, k, a, lda, b, ldb, ldd):
    """BLAS ``D = op(A) op(B)`` on flat column-major buffers, by its definition (``A`` is stored ``lda`` apart, per column)."""

    def op(buf, ld, trans, i, j):  # element (i, j) of op(X)
        return buf[j + i * ld] if trans == "T" else buf[i + j * ld]

    out = [0] * (ldd * n)
    for j in range(n):
        for i in range(m):
            out[i + j * ldd] = sum(op(a, lda, transa, i, l) * op(b, ldb, transb, l, j) for l in range(k))
    return out


@pytest.mark.parametrize("m,n,k", [(3, 4, 5), (5, 2, 3), (1, 7, 2), (4, 1, 6), (2, 2, 2), (3, 5, 4)])
@pytest.mark.parametrize("bias", [False, True])
def test_describe_is_a_gemm_that_computes_a_times_b_transposed(m, n, k, bias):
    """Lay tilefish's row-major ``A[m, k]`` and ``B[n, k]`` out in flat buffers, run the BLAS gemm that ``describe`` names on
    them as column-major matrices, and the flat result is tilefish's row-major ``D[m, n]`` (+ ``bias[n]``)."""
    a_rows = [[(7 * i + 3 * l + 1) % 11 - 5 for l in range(k)] for i in range(m)]  # all different, with signs
    b_rows = [[(5 * j + 2 * l + 3) % 13 - 6 for l in range(k)] for j in range(n)]
    bias_row = [100 * (j + 1) for j in range(n)]
    flat_a = [x for row in a_rows for x in row]  # row-major: element (i, l) at i * k + l
    flat_b = [x for row in b_rows for x in row]
    d = L.describe(m, n, k, bias)
    flat_d = _col_major_gemm(
        transa=d["transa"], transb=d["transb"], m=d["cublas_m"], n=d["cublas_n"], k=d["cublas_k"],
        a=flat_b, lda=d["lda"], b=flat_a, ldb=d["ldb"], ldd=d["ldd"],
    )  # fmt: skip
    if bias:
        assert d["bias_length"] == d["cublas_m"]
        # the bias is added along the rows of the column-major D: element (r, c) is at r + c * ldd
        for c in range(d["cublas_n"]):
            for r in range(d["cublas_m"]):
                flat_d[r + c * d["ldd"]] += bias_row[r]
    expected = [sum(a_rows[i][l] * b_rows[j][l] for l in range(k)) + (bias_row[j] if bias else 0) for i in range(m) for j in range(n)]
    assert flat_d == expected  # the flat buffer is D[m, n] row-major: element (i, j) at i * n + j


def test_the_wrong_translations_are_caught_by_that_check():
    """The check above is not a tautology: a swapped operand order or a missing transpose gives other numbers."""
    m, n, k = 3, 4, 5
    flat_a = [(7 * i + 3 * l + 1) % 11 - 5 for i in range(m) for l in range(k)]
    flat_b = [(5 * j + 2 * l + 3) % 13 - 6 for j in range(n) for l in range(k)]
    expected = [sum(flat_a[i * k + l] * flat_b[j * k + l] for l in range(k)) for i in range(m) for j in range(n)]
    right = dict(transa="T", transb="N", m=n, n=m, k=k, a=flat_b, lda=k, b=flat_a, ldb=k, ldd=n)
    assert _col_major_gemm(**right) == expected
    assert _col_major_gemm(**{**right, "a": flat_a, "b": flat_b, "m": m, "n": n, "ldd": m}) != expected  # operands swapped
    assert _col_major_gemm(**{**right, "transa": "N", "m": n, "lda": n, "ldd": n}) != expected  # no transpose of the weight


def test_describe_is_pure(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("describe must not load the library")

    monkeypatch.setattr(L, "_lib", boom)
    monkeypatch.setattr(L, "_torch", boom)
    L.describe(64, 1024, 1024, True)


@pytest.mark.parametrize(
    "args",
    [(0, 1, 1, False), (1, 0, 1, False), (1, 1, 0, False), (-1, 1, 1, False), (1, 2**31, 1, False), (1, 1, 2**31, True), (2**31, 1, 1, True)],
)
def test_describe_validates(args):
    with pytest.raises(TilefishRuntimeError):
        L.describe(*args)


# -- the descriptors and the calls: against a recording stand-in ---------------------------------------------------------------


def _value(x):
    """What a ctypes argument holds: a ``byref`` its target's value, a ``c_void_p`` its address, anything else itself."""
    if hasattr(x, "_obj"):
        x = x._obj
    return x.value if hasattr(x, "value") else x


class RecordingLib:
    """Stands in for ``_Library``: records every call with its arguments and hands out numbered handles."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on  # (function name, 1-based occurrence)
        self.seen = {}
        self.next_handle = 100

    def call(self, name, *args):
        self.seen[name] = self.seen.get(name, 0) + 1
        if self.fail_on == (name, self.seen[name]):
            raise TilefishRuntimeError(f"cuBLASLt call {name} failed: pretend")
        self.calls.append((name, args))
        if name.endswith("Create"):
            args[0]._obj.value = self.next_handle
            self.next_handle += 1

    def status_name(self, status):
        return f"status {status}"

    def named(self, name):
        return [[_value(a) for a in args] for n, args in self.calls if n == name]


def test_the_problem_is_a_tn_gemm_with_the_layouts_of_the_translation():
    lib = RecordingLib()
    with L._Problem(lib, 64, 3072, 1024, True, 0x1000, 8_519_680):
        pass
    desc_create = lib.named("cublasLtMatmulDescCreate")
    assert [a[1:] for a in desc_create] == [[68, 0]]  # CUBLAS_COMPUTE_32F, CUDA_R_32F
    layouts = [a[1:] for a in lib.named("cublasLtMatrixLayoutCreate")]  # type, rows, cols, ld
    assert layouts == [
        [14, 1024, 3072, 1024],  # the A operand: tilefish's B[3072, 1024] is 1024 x 3072 column-major, lda = 1024
        [14, 1024, 64, 1024],  # the B operand: tilefish's A[64, 1024] is 1024 x 64 column-major, ldb = 1024
        [14, 3072, 64, 3072],  # D (and C): tilefish's D[64, 3072] is 3072 x 64 column-major, ldd = 3072
    ]
    attrs = {a[1]: a[2] for a in lib.named("cublasLtMatmulDescSetAttribute")}
    assert attrs == {
        3: 1,  # CUBLASLT_MATMUL_DESC_TRANSA = CUBLAS_OP_T
        4: 0,  # CUBLASLT_MATMUL_DESC_TRANSB = CUBLAS_OP_N
        7: 4,  # CUBLASLT_MATMUL_DESC_EPILOGUE = CUBLASLT_EPILOGUE_BIAS
        26: 14,  # CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = CUDA_R_16BF
        8: 0x1000,  # CUBLASLT_MATMUL_DESC_BIAS_POINTER
    }
    sizes = {a[1]: a[3] for a in lib.named("cublasLtMatmulDescSetAttribute")}
    assert sizes == {3: 4, 4: 4, 7: 4, 26: 4, 8: 8}  # int32, int32, uint32, int32, pointer
    prefs = {a[1]: a[2] for a in lib.named("cublasLtMatmulPreferenceSetAttribute")}
    assert prefs == {
        1: 8_519_680,  # MAX_WORKSPACE_BYTES
        3: 2,  # REDUCTION_SCHEME_MASK: split-K only with f32 partial sums
        5: 16, 6: 16, 7: 16, 8: 16,  # MIN_ALIGNMENT_{A,B,C,D}_BYTES
    }  # fmt: skip
    pref_sizes = {a[1]: a[3] for a in lib.named("cublasLtMatmulPreferenceSetAttribute")}
    assert pref_sizes == {1: 8, 3: 4, 5: 4, 6: 4, 7: 4, 8: 4}  # uint64, uint32 x 5


def test_the_problem_without_a_bias_sets_no_bias_and_the_default_epilogue():
    lib = RecordingLib()
    with L._Problem(lib, 8, 16, 32, False, 0, None):
        pass
    attrs = {a[1]: a[2] for a in lib.named("cublasLtMatmulDescSetAttribute")}
    assert attrs == {3: 1, 4: 0, 7: 1}  # CUBLASLT_EPILOGUE_DEFAULT; no bias pointer, no bias type
    assert lib.named("cublasLtMatmulPreferenceCreate") == []
    assert [a[1:] for a in lib.named("cublasLtMatrixLayoutCreate")] == [[14, 32, 16, 32], [14, 32, 8, 32], [14, 16, 8, 16]]


def test_the_problem_destroys_what_it_created_in_reverse_order():
    lib = RecordingLib()
    with L._Problem(lib, 8, 16, 32, True, 16, 100):
        assert not any(n.endswith("Destroy") for n, _ in lib.calls)
    created = [(n, _value(a[0])) for n, a in lib.calls if n.endswith("Create")]
    destroyed = [(n, _value(a[0])) for n, a in lib.calls if n.endswith("Destroy")]
    assert [n for n, _ in created] == ["cublasLtMatmulDescCreate", "cublasLtMatrixLayoutCreate", "cublasLtMatrixLayoutCreate", "cublasLtMatrixLayoutCreate", "cublasLtMatmulPreferenceCreate"]
    assert [h for _, h in destroyed] == [h for _, h in reversed(created)]
    assert [n.replace("Destroy", "Create") for n, _ in destroyed] == [n for n, _ in reversed(created)]


def test_the_problem_destroys_what_it_created_when_a_call_fails_midway():
    lib = RecordingLib(fail_on=("cublasLtMatrixLayoutCreate", 3))
    with pytest.raises(TilefishRuntimeError, match="cublasLtMatrixLayoutCreate"):
        L._Problem(lib, 8, 16, 32, False, 0, None)
    destroyed = [n for n, _ in lib.calls if n.endswith("Destroy")]
    assert destroyed == ["cublasLtMatrixLayoutDestroy", "cublasLtMatrixLayoutDestroy", "cublasLtMatmulDescDestroy"]


def test_an_error_in_the_body_is_not_hidden_by_the_destroys():
    lib = RecordingLib()
    with pytest.raises(KeyError):
        with L._Problem(lib, 8, 16, 32, False, 0, None):
            raise KeyError("body")
    assert sum(n.endswith("Destroy") for n, _ in lib.calls) == 4


def test_a_failing_destroy_is_reported_when_the_body_was_fine():
    lib = RecordingLib(fail_on=("cublasLtMatmulDescDestroy", 1))
    with pytest.raises(TilefishRuntimeError, match="cublasLtMatmulDescDestroy"):
        with L._Problem(lib, 8, 16, 32, False, 0, None):
            pass
    assert sum(n.endswith("Destroy") for n, _ in lib.calls) == 3  # the others were destroyed all the same


@pytest.fixture
def fake_runtime(monkeypatch):
    """``matmul`` and ``check_algo`` on a recording library, with no device, torch or algorithm checks."""
    lib = RecordingLib()
    monkeypatch.setattr(L, "_lib", lambda: lib)
    monkeypatch.setattr(L, "_device", lambda index: 0)
    monkeypatch.setattr(L, "_handle", lambda lib_, index: ctypes.c_void_p(7))
    monkeypatch.setattr(L, "_on_device", lambda index: contextlib.nullcontext())
    monkeypatch.setattr(L, "_validate_algo", lambda lib_, handle, index, algo_hex: {})
    return lib


def _matmul_call(lib):
    (call,) = [args for name, args in lib.calls if name == "cublasLtMatmul"]
    handles = {_value(a[0]): [_value(x) for x in a[1:]] for n, a in lib.calls if n == "cublasLtMatrixLayoutCreate"}
    return call, handles


def test_matmul_hands_tilefish_b_to_cublas_as_the_a_operand(fake_runtime):
    lib = fake_runtime
    L.matmul(0x1000, 0x2000, 0x3000, 0x4000, 64, 3072, 1024, "ab" * 64, workspace_bytes=0, stream=0x55)
    call, layouts = _matmul_call(lib)
    handle, desc, alpha, a, a_desc, b, b_desc, beta, c, c_desc, d, d_desc, algo, workspace, workspace_bytes, stream = call
    assert _value(handle) == 7
    assert _value(alpha) == 1.0 and _value(beta) == 0.0
    assert _value(a) == 0x2000 and layouts[_value(a_desc)][1:] == [1024, 3072, 1024]  # tilefish B[3072, 1024]
    assert _value(b) == 0x1000 and layouts[_value(b_desc)][1:] == [1024, 64, 1024]  # tilefish A[64, 1024]
    assert _value(c) == _value(d) == 0x3000 and _value(c_desc) == _value(d_desc)
    assert layouts[_value(d_desc)][1:] == [3072, 64, 3072]
    assert bytes(algo._obj) == bytes.fromhex("ab" * 64)
    assert _value(workspace) is None and _value(workspace_bytes) == 0
    assert _value(stream) == 0x55
    attrs = {a[1]: a[2] for a in lib.named("cublasLtMatmulDescSetAttribute")}
    assert attrs[8] == 0x4000 and attrs[7] == 4  # the real bias pointer, and the bias epilogue


def test_matmul_without_a_bias_uses_the_default_epilogue(fake_runtime):
    lib = fake_runtime
    L.matmul(0x1000, 0x2000, 0x3000, None, 4, 8, 16, "00" * 64, workspace_bytes=100, stream=0)
    attrs = {a[1]: a[2] for a in lib.named("cublasLtMatmulDescSetAttribute")}
    assert attrs == {3: 1, 4: 0, 7: 1}


def test_matmul_refuses_what_the_check_refuses_and_launches_nothing(fake_runtime, monkeypatch):
    def refuse(*args):
        raise TilefishRuntimeError("not valid")

    monkeypatch.setattr(L, "_validate_algo", refuse)
    with pytest.raises(TilefishRuntimeError, match="not valid"):
        L.matmul(0x1000, 0x2000, 0x3000, None, 4, 8, 16, "00" * 64, workspace_bytes=0, stream=0)
    assert fake_runtime.named("cublasLtMatmul") == []
    assert sum(n.endswith("Destroy") for n, _ in fake_runtime.calls) == 4  # the descriptors are gone all the same


def test_matmul_refuses_an_algorithm_that_needs_more_workspace_than_recorded(fake_runtime, monkeypatch):
    original = fake_runtime.call

    def call(name, *args):
        original(name, *args)
        if name == "cublasLtMatmulAlgoCheck":
            args[-1]._obj.workspaceSize = 4096

    monkeypatch.setattr(fake_runtime, "call", call)
    with pytest.raises(TilefishRuntimeError, match="needs 4096 bytes of workspace.*more than the 4095 recorded"):
        L.matmul(0x1000, 0x2000, 0x3000, None, 4, 8, 16, "00" * 64, workspace_bytes=4095, stream=0)
    assert fake_runtime.named("cublasLtMatmul") == []
    assert L.check_algo(4, 8, 16, False, "00" * 64, workspace_bytes=4096) == 4096


def test_matmul_refuses_an_algorithm_the_library_does_not_accept_for_the_problem(fake_runtime, monkeypatch):
    original = fake_runtime.call

    def call(name, *args):
        if name == "cublasLtMatmulAlgoCheck":
            raise TilefishRuntimeError("cuBLASLt call cublasLtMatmulAlgoCheck failed: CUBLAS_STATUS_NOT_SUPPORTED (15)")
        original(name, *args)

    monkeypatch.setattr(fake_runtime, "call", call)
    with pytest.raises(TilefishRuntimeError, match="not valid for the matmul m=4 n=8 k=16 bias=False.*CUBLAS_STATUS_NOT_SUPPORTED"):
        L.check_algo(4, 8, 16, False, "00" * 64, workspace_bytes=0)


def test_check_algo_uses_a_dummy_bias_pointer_of_16_byte_alignment(fake_runtime):
    L.check_algo(4, 8, 16, True, "00" * 64, workspace_bytes=0)
    attrs = {a[1]: a[2] for a in fake_runtime.named("cublasLtMatmulDescSetAttribute")}
    assert attrs[8] == L.HEURISTIC_BIAS_POINTER == 16


# -- argument validation, before anything reaches the library ----------------------------------------------------------------


@pytest.fixture
def untouchable_library(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("the library must not be reached")

    monkeypatch.setattr(L, "_lib", boom)
    monkeypatch.setattr(L, "_torch", boom)
    monkeypatch.setattr(L, "_device", boom)


class Reached(Exception):
    pass


@pytest.fixture
def reachable_library(monkeypatch):
    def reached(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(L, "_lib", reached)


BAD_SIZES = [0, -1, 2**31, 2**40, 1.0, "1", None, True, False, 1.5]


@pytest.mark.parametrize("bad", BAD_SIZES)
@pytest.mark.parametrize("which", [0, 1, 2])
def test_pin_algo_validates_the_sizes(untouchable_library, bad, which):
    args = [8, 16, 32]
    args[which] = bad
    with pytest.raises(TilefishRuntimeError, match="must be an int|outside"):
        L.pin_algo(*args)


@pytest.mark.parametrize("bad", BAD_SIZES)
@pytest.mark.parametrize("which", [0, 1, 2])
def test_check_algo_and_matmul_validate_the_sizes(untouchable_library, bad, which):
    args = [8, 16, 32]
    args[which] = bad
    with pytest.raises(TilefishRuntimeError, match="must be an int|outside"):
        L.check_algo(*args, False, HEX, workspace_bytes=0)
    sizes = dict(zip("mnk", args))
    with pytest.raises(TilefishRuntimeError, match="must be an int|outside"):
        L.matmul(16, 32, 48, None, **sizes, algo_hex=HEX, workspace_bytes=0, stream=0)


@pytest.mark.parametrize("bias", [1, 0, None, "yes", 1.0])
def test_bias_is_a_bool(untouchable_library, bias):
    with pytest.raises(TilefishRuntimeError, match="bias must be a bool"):
        L.pin_algo(8, 16, 32, bias)
    with pytest.raises(TilefishRuntimeError, match="bias must be a bool"):
        L.check_algo(8, 16, 32, bias, HEX, workspace_bytes=0)


@pytest.mark.parametrize("bad", [-1, 2**64, 1.0, "8", None, True])
def test_the_workspace_is_a_non_negative_64_bit_int(untouchable_library, bad):
    with pytest.raises(TilefishRuntimeError, match="workspace_bytes"):
        L.pin_algo(8, 16, 32, workspace_bytes=bad)
    with pytest.raises(TilefishRuntimeError, match="workspace_bytes"):
        L.check_algo(8, 16, 32, False, HEX, workspace_bytes=bad)
    with pytest.raises(TilefishRuntimeError, match="workspace_bytes"):
        L.matmul(16, 32, 48, None, 8, 16, 32, HEX, workspace_bytes=bad, stream=0)


@pytest.mark.parametrize("bad", [-1, 2**31, 1.0, "0", True])
def test_the_device_index_is_a_non_negative_int32(untouchable_library, bad):
    with pytest.raises(TilefishRuntimeError, match="device_index"):
        L.pin_algo(8, 16, 32, device_index=bad)
    with pytest.raises(TilefishRuntimeError, match="device_index"):
        L.check_algo(8, 16, 32, False, HEX, workspace_bytes=0, device_index=bad)
    with pytest.raises(TilefishRuntimeError, match="device_index"):
        L.matmul(16, 32, 48, None, 8, 16, 32, HEX, workspace_bytes=0, stream=0, device_index=bad)


@pytest.mark.parametrize("bad", ["", HEX[:-2], HEX.upper(), HEX + "00", None, bytes.fromhex(HEX)])
def test_a_malformed_algorithm_is_refused_before_the_library(untouchable_library, bad):
    with pytest.raises(TilefishRuntimeError, match="lowercase hex|string"):
        L.check_algo(8, 16, 32, False, bad, workspace_bytes=0)
    with pytest.raises(TilefishRuntimeError, match="lowercase hex|string"):
        L.matmul(16, 32, 48, None, 8, 16, 32, bad, workspace_bytes=0, stream=0)


@pytest.mark.parametrize("which", ["A", "B", "D", "bias"])
@pytest.mark.parametrize("bad", [0, -16, 8, 17, 4, 2**64, 2**64 + 16, 16.0, "16", True])
def test_pointers_are_16_byte_aligned_non_null_addresses(untouchable_library, which, bad):
    pointers = {"A": 0x1000, "B": 0x2000, "D": 0x3000, "bias": 0x4000}
    pointers[which] = bad
    with pytest.raises(TilefishRuntimeError, match=rf"{which} pointer|must be an int|outside"):
        L.matmul(pointers["A"], pointers["B"], pointers["D"], pointers["bias"], 8, 16, 32, HEX, workspace_bytes=0, stream=0)


def test_a_missing_operand_pointer_is_refused(untouchable_library):
    for index in range(3):
        pointers = [0x1000, 0x2000, 0x3000]
        pointers[index] = None
        with pytest.raises(TilefishRuntimeError, match="pointer"):
            L.matmul(*pointers, None, 8, 16, 32, HEX, workspace_bytes=0, stream=0)


@pytest.mark.parametrize("bad", [0, 16, 128, 0x1000 + 16, 0x1000 + 255, -256, 2**64, 256.0, "256", True])
def test_a_workspace_the_caller_gives_is_256_byte_aligned(untouchable_library, bad):
    """What the Rust host requires of a caller's workspace (cuBLASLt asks for 16 bytes at least and recommends 256), checked before the library is touched."""
    with pytest.raises(TilefishRuntimeError, match=r"workspace pointer"):
        L.matmul(0x1000, 0x2000, 0x3000, None, 8, 16, 32, HEX, workspace_bytes=1024, stream=0, workspace_ptr=bad)


def test_a_workspace_aligned_to_256_bytes_reaches_the_library(reachable_library):
    with pytest.raises(Reached):
        L.matmul(0x1000, 0x2000, 0x3000, None, 8, 16, 32, HEX, workspace_bytes=1024, stream=0, workspace_ptr=0x10100)
    with pytest.raises(TilefishRuntimeError, match=r"the workspace pointer 0x10010 is not 256-byte aligned"):
        L.matmul(0x1000, 0x2000, 0x3000, None, 8, 16, 32, HEX, workspace_bytes=1024, stream=0, workspace_ptr=0x10010)  # 16 bytes was enough before


def test_the_message_of_a_misaligned_pointer_says_which_and_why(untouchable_library):
    with pytest.raises(TilefishRuntimeError, match=r"the D pointer 0x3008 is not 16-byte aligned"):
        L.matmul(0x1000, 0x2000, 0x3008, None, 8, 16, 32, HEX, workspace_bytes=0, stream=0)


@pytest.mark.parametrize("bad", [-1, 2**64, 1.0, "0", None, True])
def test_the_stream_is_a_non_negative_64_bit_int(untouchable_library, bad):
    with pytest.raises(TilefishRuntimeError, match="stream"):
        L.matmul(16, 32, 48, None, 8, 16, 32, HEX, workspace_bytes=0, stream=bad)


def test_valid_arguments_reach_the_library(reachable_library):
    with pytest.raises(Reached):
        L.pin_algo(8, 16, 32)
    with pytest.raises(Reached):
        L.pin_algo(1, 1, 1, True, workspace_bytes=0, device_index=0)
    with pytest.raises(Reached):
        L.pin_algo(2**31 - 1, 2**31 - 1, 2**31 - 1)
    with pytest.raises(Reached):
        L.check_algo(8, 16, 32, True, HEX, workspace_bytes=2**64 - 1)
    with pytest.raises(Reached):
        L.matmul(16, 32, 48, 64, 8, 16, 32, HEX, workspace_bytes=0, stream=0)
    with pytest.raises(Reached):
        L.matmul(2**64 - 16, 2**63, 16, None, 8, 16, 32, HEX, workspace_bytes=0, stream=2**64 - 1)


# -- finding and loading the library ------------------------------------------------------------------------------------------


def _write_maps(tmp_path, paths):
    lines = ["7f0000000000-7f0000001000 r-xp 00000000 08:01 1234 /usr/lib/libc.so.6", "7f0000002000-7f0000003000 rw-p 00000000 00:00 0"]
    for i, path in enumerate(paths):
        lines.append(f"7f00{i:02x}000000-7f00{i:02x}001000 r--p 00000000 08:01 {i + 10} {path}")
        lines.append(f"7f00{i:02x}002000-7f00{i:02x}003000 r-xp 00001000 08:01 {i + 10} {path}")
    maps = tmp_path / "maps"
    maps.write_text("\n".join(lines) + "\n")
    return maps


def test_mapped_paths_reads_the_distinct_libcublaslt_files(tmp_path, monkeypatch):
    real = tmp_path / "wheel" / "libcublasLt.so.12.8.4.1"
    real.parent.mkdir()
    real.write_bytes(b"")
    link_dir = tmp_path / "link"
    link_dir.mkdir()
    (link_dir / "libcublasLt.so.12").symlink_to(real)
    other = tmp_path / "other dir" / "libcublasLt.so"
    other.parent.mkdir()
    other.write_bytes(b"")
    lookalikes = [tmp_path / "libcublas.so.12", tmp_path / "libcublasLt.so.12.dbg", tmp_path / "xlibcublasLt.so.12", tmp_path / "libcublasLt.a"]
    maps = _write_maps(tmp_path, [real, link_dir / "libcublasLt.so.12", real, other, *lookalikes, f"{tmp_path}/gone/libcublasLt.so.11 (deleted)"])
    monkeypatch.setattr(L, "_MAPS", str(maps))
    assert L._mapped_paths() == sorted(
        {os.path.realpath(real), os.path.realpath(other), os.path.realpath(tmp_path / "gone" / "libcublasLt.so.11")}
    )


def test_mapped_paths_is_empty_without_proc_maps(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "_MAPS", str(tmp_path / "no-such-file"))
    assert L._mapped_paths() == []


def test_the_mapped_library_is_used_and_nothing_is_triggered_or_loaded(monkeypatch):
    monkeypatch.setattr(L, "_mapped_paths", lambda: ["/torch/libcublasLt.so.12"])
    monkeypatch.setattr(L, "_trigger_torch_load", lambda: pytest.fail("torch has a cuBLASLt mapped already"))
    monkeypatch.setattr(L, "_wheel_path", lambda: pytest.fail("the wheel must not be opened beside a mapped library"))
    assert L.library_path() == "/torch/libcublasLt.so.12"


def test_torch_is_made_to_load_its_own_before_the_wheel_is_considered(monkeypatch):
    state = {"mapped": [], "triggered": 0}

    def trigger():
        state["triggered"] += 1
        state["mapped"] = ["/torch/libcublasLt.so.12"]

    monkeypatch.setattr(L, "_mapped_paths", lambda: list(state["mapped"]))
    monkeypatch.setattr(L, "_trigger_torch_load", trigger)
    monkeypatch.setattr(L, "_wheel_path", lambda: pytest.fail("torch mapped its own, the wheel is not to be opened"))
    assert L.library_path() == "/torch/libcublasLt.so.12"
    assert state["triggered"] == 1


def test_the_wheel_is_the_last_resort(monkeypatch):
    triggered = []
    monkeypatch.setattr(L, "_mapped_paths", lambda: [])
    monkeypatch.setattr(L, "_trigger_torch_load", lambda: triggered.append(1))
    monkeypatch.setattr(L, "_wheel_path", lambda: "/wheel/libcublasLt.so.12")
    assert L.library_path() == "/wheel/libcublasLt.so.12"
    assert triggered == [1]  # torch was given its chance first


def test_two_different_mapped_copies_are_an_error_naming_both(monkeypatch):
    monkeypatch.setattr(L, "_mapped_paths", lambda: ["/a/libcublasLt.so.12", "/b/libcublasLt.so.12"])
    with pytest.raises(TilefishRuntimeError, match=r"2 different libcublasLt files \(/a/libcublasLt.so.12, /b/libcublasLt.so.12\)"):
        L.library_path()


def test_no_cublaslt_anywhere_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(L, "_library", None)
    monkeypatch.setattr(L, "_mapped_paths", lambda: [])
    monkeypatch.setattr(L, "_wheel_path", lambda: None)
    monkeypatch.setattr(L, "_trigger_torch_load", lambda: None)
    calls = [
        L.library_path,
        L.version,
        lambda: L.pin_algo(8, 16, 32),
        lambda: L.check_algo(8, 16, 32, False, HEX, workspace_bytes=0),
        lambda: L.matmul(16, 32, 48, None, 8, 16, 32, HEX, workspace_bytes=0, stream=0),
        lambda: L.algo_info(HEX),
    ]
    for call in calls:
        with pytest.raises(TilefishRuntimeError, match=r"cuBLASLt is not available.*nvidia-cublas") as info:
            call()
        assert isinstance(info.value, RuntimeError)


def test_wheel_path_finds_the_library_of_the_nvidia_cublas_package(tmp_path, monkeypatch):
    lib_dir = tmp_path / "nvidia" / "cublas" / "lib"
    lib_dir.mkdir(parents=True)
    for name in ["libcublas.so.12", "libcublasLt.so.12", "libcublasLt.so.11", "libcublasLt.so.12.dbg", "libnvblas.so.12"]:
        (lib_dir / name).write_bytes(b"")
    spec = types.SimpleNamespace(submodule_search_locations=[str(tmp_path / "elsewhere"), str(lib_dir)])
    monkeypatch.setattr("importlib.util.find_spec", lambda name: spec if name == "nvidia.cublas.lib" else None)
    assert L._wheel_path() == os.path.realpath(lib_dir / "libcublasLt.so.12")  # the highest version


def test_wheel_path_is_none_without_the_package(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert L._wheel_path() is None

    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("importlib.util.find_spec", missing)
    assert L._wheel_path() is None
    monkeypatch.setattr("importlib.util.find_spec", lambda name: types.SimpleNamespace(submodule_search_locations=None))
    assert L._wheel_path() is None


def test_the_trigger_runs_a_tiny_bf16_matmul_when_torch_has_cuda(monkeypatch):
    log = []
    fake = types.SimpleNamespace(
        bfloat16="bf16",
        ones=lambda *shape, **kwargs: log.append(("ones", shape, kwargs)) or "tensor",
        mm=lambda a, b: log.append(("mm", a, b)),
        cuda=types.SimpleNamespace(is_available=lambda: True, synchronize=lambda: log.append("sync")),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    L._trigger_torch_load()
    assert log == [("ones", (16, 16), {"dtype": "bf16", "device": "cuda"}), ("mm", "tensor", "tensor"), "sync"]


def test_the_trigger_does_nothing_without_cuda_and_never_raises(monkeypatch):
    log = []
    no_cuda = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False), ones=lambda *a, **k: log.append("ones"))
    monkeypatch.setitem(sys.modules, "torch", no_cuda)
    L._trigger_torch_load()
    assert log == []

    def broken():
        raise RuntimeError("CUDA initialization: unknown error")

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=broken)))
    L._trigger_torch_load()
    monkeypatch.setitem(sys.modules, "torch", None)  # an import of it raises ImportError
    L._trigger_torch_load()


def test_the_torch_helpers_say_what_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(TilefishRuntimeError, match="needs torch with CUDA"):
        L._torch()
    no_cuda = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", no_cuda)
    with pytest.raises(TilefishRuntimeError, match="needs a CUDA device"):
        L._device(None)
    two = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2, current_device=lambda: 1))
    monkeypatch.setitem(sys.modules, "torch", two)
    assert L._device(None) == 1
    assert L._device(0) == 0
    with pytest.raises(TilefishRuntimeError, match="device_index 2 is out of range, torch sees 2"):
        L._device(2)


FUNCTIONS = {
    # name: number of parameters in the header (cublasLt.h)
    "cublasLtGetVersion": 0,
    "cublasLtGetStatusName": 1,
    "cublasLtCreate": 1,
    "cublasLtDestroy": 1,
    "cublasLtMatmulDescCreate": 3,
    "cublasLtMatmulDescDestroy": 1,
    "cublasLtMatmulDescSetAttribute": 4,
    "cublasLtMatrixLayoutCreate": 5,
    "cublasLtMatrixLayoutDestroy": 1,
    "cublasLtMatmulPreferenceCreate": 1,
    "cublasLtMatmulPreferenceDestroy": 1,
    "cublasLtMatmulPreferenceSetAttribute": 4,
    "cublasLtMatmulAlgoGetHeuristic": 10,
    "cublasLtMatmulAlgoCheck": 8,
    "cublasLtMatmul": 16,
    "cublasLtMatmulAlgoConfigGetAttribute": 5,
    "cublasLtMatmulAlgoCapGetAttribute": 5,
    "cublasLtMatmulAlgoConfigSetAttribute": 4,
    "cublasLtMatmulAlgoInit": 9,
}


def _fake_cdll():
    return mock.MagicMock(name="CDLL")


def test_the_library_gets_a_prototype_for_every_function(tmp_path, monkeypatch):
    path = tmp_path / "libcublasLt.so.12"
    path.write_bytes(b"")
    cdll = _fake_cdll()
    monkeypatch.setattr(L.ctypes, "CDLL", lambda p: cdll)
    monkeypatch.setattr(L, "_MAPS", str(_write_maps(tmp_path, [path])))
    monkeypatch.setattr(L, "_mapped_paths", lambda: [os.path.realpath(path)])
    lib = L._Library(str(path))
    assert lib.path == os.path.realpath(path)
    assert set(FUNCTIONS) == set(L._functions())
    for name, count in FUNCTIONS.items():
        fn = getattr(cdll, name)
        assert len(fn.argtypes) == count, name
        expected = ctypes.c_size_t if name == "cublasLtGetVersion" else ctypes.c_char_p if name == "cublasLtGetStatusName" else ctypes.c_int
        assert fn.restype is expected, name


def test_a_second_copy_of_the_library_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "libcublasLt.so.12"
    path.write_bytes(b"")
    monkeypatch.setattr(L.ctypes, "CDLL", lambda p: _fake_cdll())
    monkeypatch.setattr(L, "_MAPS", str(_write_maps(tmp_path, [path])))
    monkeypatch.setattr(L, "_mapped_paths", lambda: ["/torch/libcublasLt.so.12", os.path.realpath(path)])
    with pytest.raises(TilefishRuntimeError, match="must not map a second libcublasLt"):
        L._Library(str(path))
    monkeypatch.setattr(L, "_mapped_paths", lambda: ["/torch/libcublasLt.so.12"])  # opened something else than what is mapped
    with pytest.raises(TilefishRuntimeError, match="must not map a second libcublasLt"):
        L._Library(str(path))


def test_a_library_that_cannot_be_opened_is_a_clear_error(tmp_path, monkeypatch):
    def fail(path):
        raise OSError("cannot open shared object file")

    monkeypatch.setattr(L.ctypes, "CDLL", fail)
    with pytest.raises(TilefishRuntimeError, match=r"cuBLASLt \(.*libcublasLt.so.12\) cannot be loaded: cannot open shared object file"):
        L._Library(str(tmp_path / "libcublasLt.so.12"))


def test_a_library_without_a_function_is_too_old(tmp_path, monkeypatch):
    path = tmp_path / "libcublasLt.so.12"
    path.write_bytes(b"")

    class Old:
        def __getattr__(self, name):
            if name == "cublasLtMatmulAlgoCapGetAttribute":
                raise AttributeError(name)
            return mock.MagicMock()

    monkeypatch.setattr(L.ctypes, "CDLL", lambda p: Old())
    monkeypatch.setattr(L, "_MAPS", str(tmp_path / "none"))
    with pytest.raises(TilefishRuntimeError, match="has no cublasLtMatmulAlgoCapGetAttribute: it is too old"):
        L._Library(str(path))


def _library_with(tmp_path, monkeypatch, **returns):
    path = tmp_path / "libcublasLt.so.12"
    path.write_bytes(b"")
    cdll = _fake_cdll()
    for name, value in returns.items():
        getattr(cdll, name).return_value = value
    monkeypatch.setattr(L.ctypes, "CDLL", lambda p: cdll)
    monkeypatch.setattr(L, "_MAPS", str(tmp_path / "none"))
    return L._Library(str(path))


def test_an_error_status_names_the_call_and_the_status(tmp_path, monkeypatch):
    lib = _library_with(tmp_path, monkeypatch, cublasLtMatmulDescCreate=7, cublasLtMatmulAlgoCheck=15, cublasLtMatmul=0)
    lib.call("cublasLtMatmul", 1, 2)  # success is silent
    with pytest.raises(TilefishRuntimeError, match=r"cuBLASLt call cublasLtMatmulDescCreate failed: CUBLAS_STATUS_INVALID_VALUE \(7\)"):
        lib.call("cublasLtMatmulDescCreate", ctypes.byref(ctypes.c_void_p()), 68, 0)
    with pytest.raises(TilefishRuntimeError, match=r"cublasLtMatmulAlgoCheck failed: CUBLAS_STATUS_NOT_SUPPORTED \(15\)"):
        lib.call("cublasLtMatmulAlgoCheck")


def test_an_unknown_status_is_named_by_the_library_or_by_its_number(tmp_path, monkeypatch):
    lib = _library_with(tmp_path, monkeypatch, cublasLtCreate=99, cublasLtGetStatusName=b"CUBLAS_STATUS_FUTURE")
    with pytest.raises(TilefishRuntimeError, match=r"cublasLtCreate failed: CUBLAS_STATUS_FUTURE \(99\)"):
        lib.call("cublasLtCreate")
    lib.cublasLtGetStatusName.return_value = None
    with pytest.raises(TilefishRuntimeError, match=r"cublasLtCreate failed: status 99"):
        lib.call("cublasLtCreate")


def test_the_status_codes_are_the_ones_of_the_header():
    assert L._STATUS_NAMES == {
        0: "CUBLAS_STATUS_SUCCESS", 1: "CUBLAS_STATUS_NOT_INITIALIZED", 3: "CUBLAS_STATUS_ALLOC_FAILED", 7: "CUBLAS_STATUS_INVALID_VALUE",
        8: "CUBLAS_STATUS_ARCH_MISMATCH", 11: "CUBLAS_STATUS_MAPPING_ERROR", 13: "CUBLAS_STATUS_EXECUTION_FAILED",
        14: "CUBLAS_STATUS_INTERNAL_ERROR", 15: "CUBLAS_STATUS_NOT_SUPPORTED", 16: "CUBLAS_STATUS_LICENSE_ERROR",
    }  # fmt: skip


def test_the_enum_values_are_the_ones_of_the_cublas_12_8_headers():
    # cublas_api.h, library_types.h, cublasLt.h of the nvidia-cublas 12.8 wheel
    assert (L.CUBLAS_OP_N, L.CUBLAS_OP_T) == (0, 1)
    assert L.CUBLAS_COMPUTE_32F == 68
    assert (L.CUDA_R_32F, L.CUDA_R_16BF) == (0, 14)
    assert (L.CUBLASLT_MATMUL_DESC_TRANSA, L.CUBLASLT_MATMUL_DESC_TRANSB) == (3, 4)
    assert (L.CUBLASLT_MATMUL_DESC_EPILOGUE, L.CUBLASLT_MATMUL_DESC_BIAS_POINTER, L.CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE) == (7, 8, 26)
    assert (L.CUBLASLT_EPILOGUE_DEFAULT, L.CUBLASLT_EPILOGUE_BIAS) == (1, 4)
    assert (L.CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, L.CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK) == (1, 3)
    assert (
        L.CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, L.CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        L.CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, L.CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
    ) == (5, 6, 7, 8)  # fmt: skip
    assert (L.CUBLASLT_REDUCTION_SCHEME_NONE, L.CUBLASLT_REDUCTION_SCHEME_INPLACE) == (0, 1)
    assert (L.CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE, L.CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE) == (2, 4)
    assert {name: attr for name, (attr, _) in L._ALGO_CONFIG.items()} == {
        "id": 0, "tile_id": 1, "splitk_num": 2, "reduction_scheme": 3, "cta_swizzling": 4, "custom_option": 5,
        "stages_id": 6, "inner_shape_id": 7, "cluster_shape_id": 8,
    }  # fmt: skip
    assert L.DEFAULT_WORKSPACE_BYTES == 4096 * 1024 * 2 + 16 * 1024 * 8 == 8_519_680


def test_the_structs_have_the_size_of_the_header():
    assert ctypes.sizeof(L._Algo) == 64 == L.ALGO_BYTES  # uint64_t data[8]
    assert ctypes.sizeof(L._HeuristicResult) == 96  # 64 + size_t + int + float + int[4]


def test_the_version_is_what_the_library_says(tmp_path, monkeypatch):
    lib = _library_with(tmp_path, monkeypatch, cublasLtGetVersion=120804)
    monkeypatch.setattr(L, "_library", lib)
    assert L.version() == 120804 and type(L.version()) is int


def test_the_library_is_loaded_once(tmp_path, monkeypatch):
    path = tmp_path / "libcublasLt.so.12"
    path.write_bytes(b"")
    opened = []

    def cdll(p):
        opened.append(p)
        return _fake_cdll()

    monkeypatch.setattr(L, "_library", None)
    monkeypatch.setattr(L.ctypes, "CDLL", cdll)
    monkeypatch.setattr(L, "_MAPS", str(tmp_path / "none"))
    monkeypatch.setattr(L, "library_path", lambda: str(path))
    first = L._lib()
    assert L._lib() is first and opened == [str(path)]


# -- no torch at import time -------------------------------------------------------------------------------------------------


def test_importing_the_module_does_not_import_torch():
    code = "import sys; import tilefish._cublaslt; assert 'torch' not in sys.modules, 'torch was imported'"
    env = dict(os.environ)
    src = os.path.dirname(os.path.dirname(os.path.abspath(L.__file__)))
    env["PYTHONPATH"] = os.pathsep.join([src, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stderr
