"""Loading a cubin and launching it through the CUDA driver API: what a C++ or Rust host does.

``cuModuleLoadData`` on the cubin bytes, ``cuModuleGetFunction`` on the manifest's symbol,
``cuFuncSetAttribute`` when the kernel needs more than the default 48 KiB of dynamic shared memory, and
``cuLaunchKernel`` with ``kernelParams`` laid out exactly as the manifest's ABI says: the non-constexpr
arguments in declaration order (a pointer as a 64-bit value, an ``i32`` as a 32-bit one), then the two
scratch slots as null 64-bit pointers. Nothing else is set up, so a binary that wants more (scratch memory,
programmatic dependent launch, clusters, a cooperative launch) is refused by :func:`load` before it runs.

A launch is checked before it reaches the driver, because a wrong one does not fail cleanly: a missing
argument reads whatever follows the array, an ``i32`` out of range wraps, a grid beyond the device's limits is an
error, and too little shared memory makes the kernel fault and poison the context (every later call fails). So
:class:`LoadedKernel` checks the arguments against the ABI (count, kinds, ranges), the grid against the device's
maximum grid dimensions, and the shared memory against what the driver enforces: the function's static shared memory
(``CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES``, as the driver reports it once the module is loaded) plus the dynamic shared
memory of the manifest must not exceed the device's opt-in maximum per block (``MAX_SHARED_MEMORY_PER_BLOCK_OPTIN``).
That is measured, not assumed: a Triton kernel whose ELF has a ``.nv.shared.<symbol>`` section of 1024 bytes reports 0
static bytes and the driver takes a dynamic size of exactly the opt-in maximum; a hand-written kernel with 4096 bytes of
static shared memory reports 4096 and takes the opt-in maximum minus 4096, and not a byte more.

The driver binding is ``cuda-python`` (``cuda.bindings.driver``), imported when first needed.
"""

from __future__ import annotations

import ctypes
import json
import threading
from typing import TYPE_CHECKING, Any, Sequence

from .errors import TilefishRuntimeError

if TYPE_CHECKING:
    from ._build import BuiltKernel

DEFAULT_SHARED_LIMIT = 48 * 1024  # dynamic shared memory above this needs cuFuncSetAttribute
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1
UINT64_MAX = (1 << 64) - 1

_lock = threading.Lock()
_loaded: "dict[tuple, LoadedKernel]" = {}


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


def load_key(built: "BuiltKernel", device_index: int) -> "tuple[Any, ...]":
    """What makes two loads the same: the cubin, the device, and everything the launch is made of.

    Two entries with one cubin but another symbol, shared memory, launch attributes, block or ABI are different
    kernels as far as a launch is concerned (the ABI is what the arguments of a launch are checked and packed by, and what
    the loaded function's parameters are checked against), so a cached one is never reused for them.
    """
    launch = built.launch.to_dict()
    return (
        built.sha256,
        device_index,
        built.symbol,
        built.shared,
        tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in sorted(launch.items())),
        tuple(built.spec.block),
        json.dumps(built.abi(), sort_keys=True),
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
        (device,) = _check(drv.cuCtxGetDevice(), "cuCtxGetDevice")
        if int(device) != device_index:
            raise TilefishRuntimeError(
                f"'{built.stem}' ({built.arch}) is to be loaded for device {device_index}, but the current CUDA context is on "
                f"device {int(device)}: make that device current first (with torch.cuda.device({device_index}))"
            )
        self.built = built
        self.device_index = device_index
        self.grid_limits = tuple(
            int(
                _check(
                    drv.cuDeviceGetAttribute(getattr(drv.CUdevice_attribute, f"CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_{axis}"), device),
                    f"cuDeviceGetAttribute(MAX_GRID_DIM_{axis})",
                )[0]
            )
            for axis in "XYZ"
        )
        (optin,) = _check(
            drv.cuDeviceGetAttribute(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device),
            "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)",
        )
        (self.module,) = _check(drv.cuModuleLoadData(built.cubin), f"cuModuleLoadData({built.stem})")
        try:
            (self.function,) = _check(
                drv.cuModuleGetFunction(self.module, built.symbol.encode("utf-8")), f"cuModuleGetFunction({built.symbol})"
            )
            (static,) = _check(
                drv.cuFuncGetAttribute(drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, self.function),
                "cuFuncGetAttribute(SHARED_SIZE_BYTES)",
            )
            self.static_shared = int(static)
            if self.static_shared + built.shared > int(optin):
                raise TilefishRuntimeError(
                    f"'{built.stem}' ({built.arch}) needs {self.static_shared} bytes of static and {built.shared} bytes of dynamic "
                    f"shared memory, more than the {int(optin)} bytes a block may use on this device"
                )
            if built.shared > DEFAULT_SHARED_LIMIT:
                attr = drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
                _check(drv.cuFuncSetAttribute(self.function, attr, built.shared), "cuFuncSetAttribute(max dynamic shared)")
            self._check_params()
        except BaseException:
            drv.cuModuleUnload(self.module)  # a kernel that is refused must not stay loaded
            raise
        self.block = tuple(built.spec.block)
        abi = built.abi()
        self._abi = abi["params"]

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

    def _check_launch(self, grid: "Sequence[int]", args: "Sequence[tuple[str, int]]") -> "tuple[int, int, int]":
        name = f"'{self.built.stem}'"
        if len(args) != len(self._abi):
            raise TilefishRuntimeError(
                f"{name} takes {len(self._abi)} arguments ({', '.join(p['name'] for p in self._abi)}), got {len(args)}"
            )
        for position, (param, arg) in enumerate(zip(self._abi, args)):
            if not isinstance(arg, tuple) or len(arg) != 2:
                raise TilefishRuntimeError(f"{name} argument {position} ('{param['name']}') must be a (kind, value) pair, got {arg!r}")
            kind, value = arg
            if kind != param["kind"]:
                raise TilefishRuntimeError(
                    f"{name} argument {position} ('{param['name']}') is a {param['kind']} in the ABI, but a {kind!r} was passed"
                )
            if isinstance(value, bool) or not isinstance(value, int):
                raise TilefishRuntimeError(f"{name} argument {position} ('{param['name']}') must be an int, got {value!r}")
            lo, hi = (0, UINT64_MAX) if kind == "pointer" else (INT32_MIN, INT32_MAX)
            if not lo <= value <= hi:
                raise TilefishRuntimeError(
                    f"{name} argument {position} ('{param['name']}') is {value}, outside the {'64-bit address' if kind == 'pointer' else '32-bit signed integer'} "
                    f"range {lo}..{hi} the binary takes"
                )
        dims = tuple(grid)
        if not 1 <= len(dims) <= 3 or any(isinstance(g, bool) or not isinstance(g, int) for g in dims):
            raise TilefishRuntimeError(f"{name}: the grid must be one to three integers, got {tuple(grid)!r}")
        padded = dims + (1,) * (3 - len(dims))
        for axis, (g, limit) in enumerate(zip(padded, self.grid_limits)):
            if not 1 <= g <= limit:
                raise TilefishRuntimeError(
                    f"{name}: the grid {dims} has {g} blocks along {'XYZ'[axis]}, but this device takes 1 to {limit} there"
                )
        return padded  # type: ignore[return-value]

    def launch(self, grid: "Sequence[int]", args: "Sequence[tuple[str, int]]", stream: int) -> None:
        """Launch on ``stream`` (a ``CUstream`` handle as an int). ``args`` are ``("pointer", address)`` or ``("scalar", i32)``
        in the manifest's order; the two scratch slots are appended here.

        The arguments are checked against the ABI first (count, kinds, ranges) and the grid against the device's
        limits; a mismatch is a :class:`~tilefish.errors.TilefishRuntimeError` and nothing is launched.
        """
        g = self._check_launch(grid, args)
        drv = _driver()
        values: list[Any] = []
        for kind, v in args:
            values.append(ctypes.c_uint64(v) if kind == "pointer" else ctypes.c_int32(v))
        values += [ctypes.c_uint64(0), ctypes.c_uint64(0)]  # global_scratch, profile_scratch
        slots = (ctypes.c_void_p * len(values))(*[ctypes.addressof(v) for v in values])
        b = self.block
        _check(
            drv.cuLaunchKernel(
                self.function, g[0], g[1], g[2], b[0], b[1], b[2], self.built.shared, drv.CUstream(stream), ctypes.addressof(slots), 0
            ),
            f"cuLaunchKernel({self.built.symbol})",
        )


def load(built: "BuiltKernel", device_index: int) -> LoadedKernel:
    """The loaded kernel for this binary on this device, loading it the first time. The device's context must be current."""
    key = load_key(built, device_index)
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
