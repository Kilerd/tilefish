"""Loading a cubin and launching it through the CUDA driver API: what a C++ or Rust host does.

``cuModuleLoadData`` on the cubin bytes, ``cuModuleGetFunction`` on the manifest's symbol,
``cuFuncSetAttribute`` when the kernel needs more than the default 48 KiB of dynamic shared memory, and
``cuLaunchKernel`` with ``kernelParams`` laid out exactly as the manifest's ABI says: the non-constexpr
arguments in declaration order (a pointer as a 64-bit value, an ``i32`` as a 32-bit one), then the two
scratch slots as null 64-bit pointers. Nothing else is set up, so a binary that wants more (scratch memory,
programmatic dependent launch, clusters, a cooperative launch) is refused by :func:`load` before it runs.

The driver binding is ``cuda-python`` (``cuda.bindings.driver``), imported when first needed.
"""

from __future__ import annotations

import ctypes
import threading
from typing import TYPE_CHECKING, Any, Sequence

from .errors import TilefishRuntimeError

if TYPE_CHECKING:
    from ._build import BuiltKernel

DEFAULT_SHARED_LIMIT = 48 * 1024  # dynamic shared memory above this needs cuFuncSetAttribute

_lock = threading.Lock()
_loaded: "dict[tuple[str, int], LoadedKernel]" = {}


def _driver() -> Any:
    try:
        from cuda.bindings import driver
    except ImportError:
        try:
            from cuda import cuda as driver  # type: ignore[no-redef]  # cuda-python before 12.8
        except ImportError as e:
            raise TilefishRuntimeError(
                "replay launches through the CUDA driver and needs cuda-python: pip install 'tilefish[gpu]'"
            ) from e
    return driver


def _check(result: Any, what: str) -> "tuple[Any, ...]":
    """Unpack a cuda-python return tuple ``(error, *values)``, raising on an error."""
    drv = _driver()
    err = result[0] if isinstance(result, tuple) else result
    if err != drv.CUresult.CUDA_SUCCESS:
        try:
            _, text = drv.cuGetErrorString(err)
            detail = text.decode("utf-8", "replace") if isinstance(text, bytes) else str(text)
        except Exception:  # noqa: BLE001 - the error text is a nicety
            detail = ""
        raise TilefishRuntimeError(f"CUDA driver call {what} failed: {getattr(err, 'name', err)} {detail}".rstrip())
    return tuple(result[1:]) if isinstance(result, tuple) else ()


def refuse_unsupported(built: "BuiltKernel") -> None:
    """Raise unless a plain ``cuLaunchKernel`` can run this binary: no scratch memory, PDL, cluster or cooperative launch."""
    blockers = built.launch.replay_blockers()
    if blockers:
        raise TilefishRuntimeError(
            f"replay cannot run '{built.stem}' ({built.arch}): the binary needs launch attributes that the replay "
            f"does not set up: {', '.join(blockers)}"
        )


class LoadedKernel:
    """A kernel function loaded from a cubin into the current CUDA context (one per device)."""

    def __init__(self, built: "BuiltKernel", device_index: int) -> None:
        refuse_unsupported(built)
        drv = _driver()
        (ctx,) = _check(drv.cuCtxGetCurrent(), "cuCtxGetCurrent")
        if not int(ctx):
            raise TilefishRuntimeError(
                "no CUDA context is current on this thread; touch the device with torch first (for example "
                "torch.zeros(1, device='cuda'))"
            )
        self.built = built
        self.device_index = device_index
        (self.module,) = _check(drv.cuModuleLoadData(built.cubin), f"cuModuleLoadData({built.stem})")
        (self.function,) = _check(
            drv.cuModuleGetFunction(self.module, built.symbol.encode("utf-8")), f"cuModuleGetFunction({built.symbol})"
        )
        if built.shared > DEFAULT_SHARED_LIMIT:
            attr = drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            _check(drv.cuFuncSetAttribute(self.function, attr, built.shared), "cuFuncSetAttribute(max dynamic shared)")
        self._check_params()
        self.block = tuple(built.spec.block)

    def _check_params(self) -> None:
        """The loaded function must take the parameters of the manifest's ABI (count and sizes), when the driver can say."""
        drv = _driver()
        query = getattr(drv, "cuFuncGetParamInfo", None)
        if query is None:
            return
        sizes: list[int] = []
        while True:
            res = query(self.function, len(sizes))
            if res[0] != drv.CUresult.CUDA_SUCCESS:
                break
            sizes.append(int(res[2]))
        abi = self.built.abi()
        expected = [p["size"] for p in abi["params"]] + [p["size"] for p in abi["scratch"]]
        if sizes != expected:
            raise TilefishRuntimeError(
                f"the loaded function '{self.built.symbol}' takes parameters of {sizes} bytes, but the ABI in the "
                f"manifest says {expected}"
            )

    def launch(self, grid: "Sequence[int]", args: "Sequence[tuple[str, int]]", stream: int) -> None:
        """Launch on ``stream`` (a ``CUstream`` handle as an int). ``args`` are ``("pointer", address)`` or ``("scalar", i32)``
        in the manifest's order; the two scratch slots are appended here."""
        drv = _driver()
        values: list[Any] = []
        for kind, v in args:
            values.append(ctypes.c_uint64(v) if kind == "pointer" else ctypes.c_int32(v))
        values += [ctypes.c_uint64(0), ctypes.c_uint64(0)]  # global_scratch, profile_scratch
        slots = (ctypes.c_void_p * len(values))(*[ctypes.addressof(v) for v in values])
        g = tuple(grid) + (1,) * (3 - len(grid))
        b = self.block
        _check(
            drv.cuLaunchKernel(
                self.function, g[0], g[1], g[2], b[0], b[1], b[2], self.built.shared, drv.CUstream(stream), ctypes.addressof(slots), 0
            ),
            f"cuLaunchKernel({self.built.symbol})",
        )


def load(built: "BuiltKernel", device_index: int) -> LoadedKernel:
    """The loaded kernel for this binary on this device, loading it the first time. The device's context must be current."""
    key = (built.sha256, device_index)
    with _lock:
        k = _loaded.get(key)
        if k is None:
            k = _loaded[key] = LoadedKernel(built, device_index)
        return k


def unload_all() -> None:
    """Unload every module this process loaded (for tests); the contexts must still be current."""
    drv = _driver()
    with _lock:
        for k in _loaded.values():
            drv.cuModuleUnload(k.module)
        _loaded.clear()
