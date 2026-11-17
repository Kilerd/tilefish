"""Why a manifest ``shared`` that is nonzero but too small cannot be detected without ``pin`` (``docs/en/manifest-schema-3.md``, section 4): the cubin does
not say how much dynamic shared memory its code addresses.

Triton gives a kernel that needs shared memory an ``extern`` array of dynamic shared memory and records its size in its metadata (``shared``), which a
build copies into the manifest and a launch passes; ptxas never sees the size. Held here, on the device's architecture, for ``tl.dot`` kernels whose
``metadata.shared`` are 16 400, 49 168 and 65 552 bytes: their ELF files have the same ``.nv.shared.<symbol>`` section (the per-block system reserve,
1024 bytes from sm_90 on) and the same ``.nv.info.<symbol>`` attributes but for the parameter records, and the driver reports the same static shared
memory for each (``CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES``). Nothing a loader can read tells the three apart, so a reader checks what the file can
say (a kernel with the section and ``shared = 0`` is refused; static + ``shared`` must fit the device) and the pin covers the rest. If a later Triton or
ptxas records the size, this test fails, and a check at load becomes possible.
"""

import struct

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("cuda.bindings.driver")

import triton.language as tl  # noqa: E402

from tilefish._elf import parse_elf  # noqa: E402


@triton.jit
def mm(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        x = tl.load(a + rm[:, None] * K + (k + rk)[None, :])
        y = tl.load(b + (k + rk)[:, None] * N + rn[None, :])
        acc += tl.dot(x, y)
    tl.store(c + rm[:, None] * N + rn[None, :], acc)


def nv_info(data: bytes) -> "list[tuple[int, object]]":
    """The attribute records of a ``.nv.info`` section: ``(attribute, value)``, the value of a sized record as bytes."""
    out, i = [], 0
    while i + 2 <= len(data):
        fmt, attr = data[i], data[i + 1]
        i += 2
        if fmt == 1:
            out.append((attr, None))
        elif fmt == 2:
            out.append((attr, data[i]))
            i += 1
        elif fmt == 3:
            out.append((attr, struct.unpack_from("<H", data, i)[0]))
            i += 2
        elif fmt == 4:
            size = struct.unpack_from("<H", data, i)[0]
            out.append((attr, bytes(data[i + 2 : i + 2 + size])))
            i += 2 + size
        else:
            break
    return out


def build(bm, bn, bk, stages):
    from triton.backends.compiler import GPUTarget

    major, minor = torch.cuda.get_device_capability()
    src = triton.compiler.ASTSource(
        fn=mm, signature={"a": "*fp16", "b": "*fp16", "c": "*fp32", "M": "i32", "N": "i32", "K": "i32"}, constexprs={"BM": bm, "BN": bn, "BK": bk}
    )
    return triton.compile(src, target=GPUTarget("cuda", major * 10 + minor, 32), options={"num_warps": 4, "num_stages": stages})


PARAMETER_INFO = 23  # EIATTR_KPARAM_INFO: one record per parameter (its offset and size), the same for kernels of one signature


def test_the_cubin_does_not_record_the_dynamic_shared_memory_its_code_addresses():
    from tilefish import _cuda

    kernels = [build(64, 64, 32, 2), build(128, 64, 64, 4), build(128, 128, 64, 3)]
    sizes = [k.metadata.shared for k in kernels]
    assert len(set(sizes)) == 3 and min(sizes) > 0, sizes  # three kernels that need three amounts of dynamic shared memory
    facts = []
    for k in kernels:
        elf = parse_elf(k.asm["cubin"])
        sym = k.metadata.name
        section = elf.section(f".nv.shared.{sym}")
        info = elf.section(f".nv.info.{sym}")
        assert section is not None and info is not None
        attrs = [(a, v) for a, v in nv_info(elf.data(info)) if a != PARAMETER_INFO]
        facts.append((section.size, attrs))
    assert facts[0] == facts[1] == facts[2], (sizes, facts)
    # the driver reports the same static shared memory for each: only the manifest's shared says how much dynamic memory a launch must pass
    statics = []
    drv = _cuda._driver()
    torch.zeros(1, device="cuda")  # the device's primary context, current on this thread
    for k in kernels:
        (mod,) = _cuda._check(drv.cuModuleLoadData(k.asm["cubin"]), "cuModuleLoadData")
        (fn,) = _cuda._check(drv.cuModuleGetFunction(mod, k.metadata.name.encode()), "cuModuleGetFunction")
        attr = drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
        (static,) = _cuda._check(drv.cuFuncGetAttribute(attr, fn), "cuFuncGetAttribute")
        statics.append(int(static))
        _cuda._check(drv.cuModuleUnload(mod), "cuModuleUnload")
    assert len(set(statics)) == 1, statics
    print(f"\n[shared] metadata.shared {sizes}: .nv.shared.<symbol> {facts[0][0]} bytes in each, the same .nv.info attributes, static shared {statics}")
