"""Loading a cubin and launching it through the CUDA driver API: what a C++ or Rust host does.

``cuModuleLoadData`` on the cubin bytes, ``cuModuleGetFunction`` on the manifest's symbol,
``cuFuncSetAttribute`` when the kernel needs more than the default 48 KiB of dynamic shared memory, and
``cuLaunchKernel`` with ``kernelParams`` laid out exactly as the manifest's ABI says: the non-constexpr
arguments in declaration order (a pointer as a 64-bit value, an ``i32`` as a 32-bit one), then the two
scratch slots: the global scratch a launch is given (a null pointer for a kernel that has none) and a null profile scratch.

**Launch shapes** (manifest schema 3, sections 4.3 to 4.6). A grid is the entry's grid of programs: a **persistent** kernel's (one entry, the
tiles) is cut to ``programs_per_sm * SMs // CTAs per program`` programs, with the SM count of the device (or of the partition :func:`sm_partition`
says) read when the kernel is loaded; a kernel of **clusters** (``num_ctas`` CTAs per program, a prebuilt binary's declared cluster) is launched
with ``cuLaunchKernelEx`` and the cluster attribute, its grid of blocks the grid of programs times the cluster; a kernel with a **global scratch**
is given ``global_scratch_size`` bytes per CTA of the launch, which the caller allocates (a plan from its arena). What neither host sets up
(profile scratch, programmatic dependent launch, a cooperative launch, a cluster of more than 8) is refused by :func:`load` before it runs.

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

import contextlib
import ctypes
import functools
import json
import threading
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from ._specvalid import decode_scalar
from ._tensormap import DESCRIPTOR_BYTES, EncodeArguments, TensorMapArg
from .errors import TilefishRuntimeError

if TYPE_CHECKING:
    from ._build import BuiltKernel

DEFAULT_SHARED_LIMIT = 48 * 1024  # dynamic shared memory above this needs cuFuncSetAttribute
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1
INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1
UINT64_MAX = (1 << 64) - 1


def _ctype(type_: str) -> Any:
    """The ctypes type that lays out one scalar launch parameter of C type ``i32``, ``i64`` or ``f32``."""
    return {"i32": ctypes.c_int32, "i64": ctypes.c_int64, "f32": ctypes.c_float}[type_]

_lock = threading.Lock()
_loaded: "dict[tuple, LoadedKernel]" = {}
_partition = threading.local()  # sm_partition(): the SMs of the partition this thread launches on
_device_sms: "dict[int, int]" = {}


@contextlib.contextmanager
def sm_partition(sms: "int | None") -> "Iterator[None]":
    """Launch on a partition of ``sms`` SMs of the device (a green context) inside the block, on this thread: a persistent kernel loaded inside it
    launches at most ``programs_per_sm * sms`` programs, and one whose bytes depend on the SM count (``tilefish.persistent(sm_count=...)``) is held
    to ``sms``. (A green context reports the device's SM count to every query, so the host says which partition it is on, as the Rust host's
    ``Manifest::with_sm_partition`` does.) ``None`` is the whole device. A plan prepared inside it is prepared for it."""
    if sms is not None and (isinstance(sms, bool) or not isinstance(sms, int) or sms < 1):
        raise TilefishRuntimeError(f"sm_partition takes the SMs of the partition (a positive int) or None, got {sms!r}")
    previous = getattr(_partition, "sms", None)
    _partition.sms = sms
    try:
        yield
    finally:
        _partition.sms = previous


def current_partition() -> "int | None":
    """The partition :func:`sm_partition` set on this thread (``None``: the whole device)."""
    return getattr(_partition, "sms", None)


def sm_count(device_index: int) -> int:
    """The SMs a launch on device ``device_index`` runs on: the partition :func:`sm_partition` says, else the device's multiprocessor count
    (``CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT``; a MIG slice has its own)."""
    sms = current_partition()
    if sms is not None:
        return sms
    with _lock:
        known = _device_sms.get(device_index)
    if known is None:
        drv = _driver()
        (device,) = _check(drv.cuDeviceGet(device_index), "cuDeviceGet")
        (known,) = _check(drv.cuDeviceGetAttribute(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device), "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)")
        known = int(known)
        with _lock:
            _device_sms[device_index] = known
    return known


def check_sm_count(built: "BuiltKernel", sms: "int | None") -> None:
    """A persistent kernel whose bytes depend on how many programs run (``tilefish.persistent(sm_count=...)``) runs on that many SMs only."""
    persistent = built.spec.persistent
    if persistent is None:
        return
    declared = persistent.get("sm_count")
    if declared is not None and declared != sms:
        raise TilefishRuntimeError(
            f"'{built.stem}' ({built.arch}) is a persistent kernel whose bytes depend on how many programs run, declared for {declared} SMs "
            f"(tilefish.persistent(sm_count={declared})), but this device{' partition' if current_partition() is not None else ''} has {sms}: "
            "declare it for this SM count (another op) to run it here"
        )


def persistent_programs(tiles: int, programs_per_sm: int, sms: int, ctas: int) -> int:
    """The programs a persistent grid of ``tiles`` launches on ``sms`` SMs: at most ``programs_per_sm`` per SM, each program ``ctas`` CTAs (a cluster),
    at least one, at most ``tiles`` (manifest schema 3, section 4.3)."""
    return max(1, min(tiles, programs_per_sm * sms // ctas))


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
    """Raise unless this host can launch the binary: no profile scratch, PDL or cooperative launch, a cluster of ``num_ctas`` CTAs (at most 8), a
    global scratch of an alignment that divides 256 bytes (:meth:`~tilefish.aot.LaunchAttributes.replay_blockers`)."""
    blockers = built.launch.replay_blockers()
    if blockers:
        raise TilefishRuntimeError(
            f"replay cannot run '{built.stem}' ({built.arch}): the binary needs launch attributes that the replay "
            f"does not set up: {', '.join(blockers)}"
        )


def load_key(built: "BuiltKernel", device_index: int, sms: "int | None" = None) -> "tuple[Any, ...]":
    """What makes two loads the same: the cubin, the device, the SMs a persistent kernel's grid is resolved for, and everything the launch is made of.

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
        tuple(built.block),  # the compiled kernel's (every partition of a warp-specialised kernel)
        json.dumps(built.abi(), sort_keys=True),
        json.dumps(built.spec.persistent, sort_keys=True),
        sms,  # (None for a kernel that is not persistent: its launch does not depend on the SM count)
    )


class LoadedKernel:
    """A kernel function loaded from a cubin into the current CUDA context (one per device)."""

    persistent: "dict[str, int] | None" = None  # a persistent kernel's grid (its spec's): None for any other
    sms: "int | None" = None  # the SMs a persistent kernel's grid is resolved for

    def __init__(self, built: "BuiltKernel", device_index: int, sms: "int | None" = None) -> None:
        refuse_unsupported(built)
        drv = _driver()
        (ctx,) = _check(drv.cuCtxGetCurrent(), "cuCtxGetCurrent")
        if not int(ctx):
            raise TilefishRuntimeError(
                "no CUDA context is current on this thread; touch the device with torch first (for example "
                "torch.zeros(1, device='cuda'))"
            )
        self.context = ctx  # what unload_all synchronizes before it unloads the module
        (device,) = _check(drv.cuCtxGetDevice(), "cuCtxGetDevice")
        if int(device) != device_index:
            raise TilefishRuntimeError(
                f"'{built.stem}' ({built.arch}) is to be loaded for device {device_index}, but the current CUDA context is on "
                f"device {int(device)}: make that device current first (with torch.cuda.device({device_index}))"
            )
        self.built = built
        self.device_index = device_index
        # a persistent kernel: the SMs its grid is resolved for (the device's, or the partition's), and the count its bytes are declared for
        self.persistent = built.spec.persistent
        self.sms = sms
        check_sm_count(built, sms)
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
        self.block = built.block  # 32 * the warps of the compiled kernel (a warp-specialised kernel's partitions in all), or a binary's declared block
        abi = built.abi()
        self._abi = abi["params"]

    @property
    def _call_params(self) -> "list[dict[str, Any]]":
        """The arguments a caller passes: every parameter but the ``fixed`` ones and the null pointers, whose value the manifest holds and the launch writes itself."""
        return [p for p in self._abi if p["kind"] not in ("fixed", "null")]

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
        from ._build import abi_slot_sizes

        expected = abi_slot_sizes(self.built.abi())  # a descriptor is its CUtensorMap, then its shape and strides
        if sizes != expected:
            raise TilefishRuntimeError(
                f"the loaded function '{self.built.symbol}' takes parameters of {sizes} bytes, but the ABI in the "
                f"manifest says {expected}"
            )

    def programs(self, grid: "Sequence[int]") -> "tuple[int, ...]":
        """The grid of programs a launch of the entry's grid ``grid`` runs: ``grid`` itself, or for a persistent kernel its one entry (the tiles)
        cut to ``programs_per_sm`` programs per SM this kernel was loaded for (:func:`persistent_programs`)."""
        if self.persistent is None:
            return tuple(grid)
        assert self.sms is not None and len(grid) == 1
        return (persistent_programs(grid[0], self.persistent["programs_per_sm"], self.sms, self.built.launch.num_ctas),)

    def launch_grid(self, grid: "Sequence[int]") -> "tuple[int, int, int]":
        """The grid of blocks of a launch of the entry's grid ``grid``: its programs (:meth:`programs`) times the cluster of a kernel of clusters."""
        return self.built.launch.cuda_grid(self.programs(grid))

    def scratch_bytes(self, grid: "Sequence[int]") -> int:
        """The bytes of global scratch a launch of the entry's grid ``grid`` is given: ``global_scratch_size`` per CTA of its programs (0: none)."""
        return self.built.launch.scratch_bytes(self.programs(grid))

    def _check_launch(self, grid: "Sequence[int]", args: "Sequence[tuple[str, int]]") -> "tuple[int, int, int]":
        name = f"'{self.built.stem}'"
        if len(args) != len(self._call_params):
            raise TilefishRuntimeError(
                f"{name} takes {len(self._call_params)} arguments ({', '.join(p['name'] for p in self._call_params)}), got {len(args)}"
            )
        for position, (param, arg) in enumerate(zip(self._call_params, args)):
            if not isinstance(arg, tuple) or len(arg) != 2:
                raise TilefishRuntimeError(f"{name} argument {position} ('{param['name']}') must be a (kind, value) pair, got {arg!r}")
            kind, value = arg
            if kind != param["kind"]:
                raise TilefishRuntimeError(
                    f"{name} argument {position} ('{param['name']}') is a {param['kind']} in the ABI, but a {kind!r} was passed"
                )
            if kind == "tensordesc":
                rank = len(param["shape"])
                if (
                    not isinstance(value, TensorMapArg)
                    or len(value.descriptor) != DESCRIPTOR_BYTES
                    or len(value.shape) != rank
                    or len(value.strides) != rank
                    or not all(0 < n <= INT32_MAX for n in value.shape)
                    or not all(0 <= s <= INT64_MAX for s in value.strides)
                ):
                    raise TilefishRuntimeError(
                        f"{name} argument {position} ('{param['name']}') is a tensor descriptor: it takes an encoded CUtensorMap of {DESCRIPTOR_BYTES} bytes "
                        f"and the {rank} sizes (i32) and strides (i64) of its tensor, got {value!r}"
                    )
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TilefishRuntimeError(f"{name} argument {position} ('{param['name']}') must be an int, got {value!r}")
            if kind == "pointer":
                lo, hi, what = 0, UINT64_MAX, "64-bit address"
            elif param["type"] == "i64":
                lo, hi, what = INT64_MIN, INT64_MAX, "64-bit signed integer"
            else:
                lo, hi, what = INT32_MIN, INT32_MAX, "32-bit signed integer"
            if not lo <= value <= hi:
                raise TilefishRuntimeError(
                    f"{name} argument {position} ('{param['name']}') is {value}, outside the {what} range {lo}..{hi} the binary takes"
                )
        dims = tuple(grid)
        if not 1 <= len(dims) <= 3 or any(isinstance(g, bool) or not isinstance(g, int) for g in dims):
            raise TilefishRuntimeError(f"{name}: the grid must be one to three integers, got {tuple(grid)!r}")
        if self.persistent is not None and len(dims) != 1:
            raise TilefishRuntimeError(f"{name}: a persistent grid has one entry (the tiles), got {dims!r}")
        for axis, (g, limit) in enumerate(zip(dims + (1,) * (3 - len(dims)), self.grid_limits)):
            if g < 1:
                raise TilefishRuntimeError(f"{name}: the grid {dims} has {g} blocks along {'XYZ'[axis]}, but this device takes 1 to {limit} there")
        blocks = self.launch_grid(dims)
        for axis, (g, limit) in enumerate(zip(blocks, self.grid_limits)):
            if not 1 <= g <= limit:
                shown = dims if blocks == dims + (1,) * (3 - len(dims)) else f"{dims} ({blocks} blocks)"
                raise TilefishRuntimeError(
                    f"{name}: the grid {shown} has {g} blocks along {'XYZ'[axis]}, but this device takes 1 to {limit} there"
                )
        return blocks

    def launch(
        self, grid: "Sequence[int]", args: "Sequence[tuple[str, Any]]", stream: int, scratch: "int | None" = None, probes: "int | None" = None
    ) -> None:
        """Launch on ``stream`` (a ``CUstream`` handle as an int). ``args`` are ``("pointer", address)``, ``("scalar", i32)`` or, for a TMA
        descriptor, ``("tensordesc", TensorMapArg)`` (its encoded CUtensorMap, then its shape and strides, each in a slot of its own) in the
        manifest's order; the two scratch slots are appended here: ``scratch`` (the address of :meth:`scratch_bytes` bytes of device memory, 256-byte
        aligned, which a kernel with a global scratch must be given and any other must not) and the profile scratch: null, or for an instrumented
        binary ``probes``, the address of its probe buffer (16-byte aligned; ``docs/instrument.md``), which no other binary is given.

        ``grid`` is the entry's grid of programs: a persistent kernel's is cut to the programs the SMs it was loaded for run, and a kernel of clusters
        is launched with ``cuLaunchKernelEx``, ``grid * cluster`` blocks in clusters of its ``cluster_dims``.

        The arguments are checked against the ABI first (count, kinds, ranges) and the grid against the device's
        limits; a mismatch is a :class:`~tilefish.errors.TilefishRuntimeError` and nothing is launched.
        """
        g = self._check_launch(grid, args)
        needs = self.built.launch.global_scratch_size > 0
        if needs != (scratch is not None):
            raise TilefishRuntimeError(
                f"'{self.built.stem}' "
                + (f"addresses {self.built.launch.global_scratch_size} bytes of global scratch per CTA: give it a scratch" if needs else "has no global scratch: give it none")
            )
        if scratch is not None and (isinstance(scratch, bool) or not isinstance(scratch, int) or not 0 < scratch <= UINT64_MAX or scratch % 256):
            raise TilefishRuntimeError(f"'{self.built.stem}': the global scratch must be a 256-byte aligned address, got {scratch!r}")
        if probes is not None and (self.built.instrument is None or isinstance(probes, bool) or not isinstance(probes, int) or not 0 < probes <= UINT64_MAX or probes % 16):
            raise TilefishRuntimeError(
                f"'{self.built.stem}': "
                + ("a probe buffer is for an instrumented binary, and this one has no probes" if self.built.instrument is None else f"the probe buffer must be a 16-byte aligned address, got {probes!r}")
            )
        drv = _driver()
        values: list[Any] = []
        addresses: list[int] = []
        passed = iter(args)

        def put(value: Any) -> None:
            values.append(value)
            addresses.append(ctypes.addressof(value))

        for p in self._abi:  # the manifest's order: the caller's arguments, with each fixed argument at its own slot
            if p["kind"] == "fixed":
                put(_ctype(p["type"])(decode_scalar(p["value"], f"fixed argument {p['name']}")))
                continue
            if p["kind"] == "null":  # a pointer every launch passes as null: the 64-bit zero at its slot
                put(ctypes.c_uint64(0))
                continue
            kind, v = next(passed)
            if kind == "tensordesc":  # the 128 bytes of the CUtensorMap (64-byte aligned, as its PTX parameter is), its shape (i32), its strides (i64)
                held = (ctypes.c_uint8 * (2 * DESCRIPTOR_BYTES))()
                at = (ctypes.addressof(held) + 63) & ~63
                ctypes.memmove(at, v.descriptor, DESCRIPTOR_BYTES)
                values.append(held)
                addresses.append(at)
                for n in v.shape:
                    put(ctypes.c_int32(n))
                for s in v.strides:
                    put(ctypes.c_int64(s))
                continue
            put(ctypes.c_uint64(v) if kind == "pointer" else _ctype(p["type"])(v))
        for k, _ in enumerate(self.built.abi()["scratch"]):  # global_scratch, profile_scratch: a Triton kernel's; a prebuilt binary has none
            given = scratch if k == 0 else probes  # (the profile scratch of an instrumented binary: its probe buffer)
            put(ctypes.c_uint64(given if given is not None else 0))
        slots = (ctypes.c_void_p * len(addresses))(*addresses)
        b = self.block
        launch = self.built.launch
        if launch.num_ctas == 1:
            _check(
                drv.cuLaunchKernel(
                    self.function, g[0], g[1], g[2], b[0], b[1], b[2], self.built.shared, drv.CUstream(stream), ctypes.addressof(slots), 0
                ),
                f"cuLaunchKernel({self.built.symbol})",
            )
            return
        # a kernel of clusters: cuLaunchKernelEx with the cluster's dimensions and the spread scheduling policy Triton's launcher asks for
        config = drv.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = g
        config.blockDimX, config.blockDimY, config.blockDimZ = b
        config.sharedMemBytes = self.built.shared
        config.hStream = drv.CUstream(stream)
        cluster = drv.CUlaunchAttribute()
        cluster.id = drv.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        cluster.value.clusterDim.x, cluster.value.clusterDim.y, cluster.value.clusterDim.z = launch.cluster_dims
        policy = drv.CUlaunchAttribute()
        policy.id = drv.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
        policy.value.clusterSchedulingPolicyPreference = drv.CUclusterSchedulingPolicy.CU_CLUSTER_SCHEDULING_POLICY_SPREAD
        config.attrs = [cluster, policy]
        config.numAttrs = 2
        _check(drv.cuLaunchKernelEx(config, self.function, ctypes.addressof(slots), 0), f"cuLaunchKernelEx({self.built.symbol})")


@functools.lru_cache(maxsize=4096)
def encode_tensor_map(arguments: EncodeArguments) -> bytes:
    """The 128 bytes of the ``CUtensorMap`` that ``cuTensorMapEncodeTiled`` encodes from ``arguments`` (every number of the call; see
    :func:`tilefish._tensormap.encode_arguments`). The map is a function of those numbers alone (the driver encodes the same bytes for the same
    call), so the last 4096 are kept: a launch of the same tensor again does not call the driver, and a launch of another address, shape or stride
    is another key, so a descriptor is never one of another binding. A call the driver refuses is a :class:`~tilefish.errors.TilefishRuntimeError`
    (the launch checks of a ``TensorDesc`` argument come first, so a refusal here is one they do not know)."""
    drv = _driver()
    u64 = getattr(drv, "cuuint64_t")
    u32 = getattr(drv, "cuuint32_t")
    a = arguments
    result = drv.cuTensorMapEncodeTiled(
        drv.CUtensorMapDataType(a.data_type),
        a.rank,
        a.address,
        [u64(v) for v in a.global_dim],
        [u64(v) for v in a.global_strides],
        [u32(v) for v in a.box_dim],
        [u32(v) for v in a.element_strides],
        drv.CUtensorMapInterleave(a.interleave),
        drv.CUtensorMapSwizzle(a.swizzle),
        drv.CUtensorMapL2promotion(a.l2_promotion),
        drv.CUtensorMapFloatOOBfill(a.oob_fill),
    )
    (tensor_map,) = _check(result, f"cuTensorMapEncodeTiled({a})")
    return ctypes.string_at(int(tensor_map.getPtr()), DESCRIPTOR_BYTES)


def load(built: "BuiltKernel", device_index: int) -> LoadedKernel:
    """The loaded kernel for this binary on this device, loading it the first time. The device's context must be current. A persistent kernel is
    loaded for the SMs of the device, or of the partition :func:`sm_partition` says (each its own load)."""
    sms = sm_count(device_index) if built.spec.persistent is not None else None
    key = load_key(built, device_index, sms)
    with _lock:
        k = _loaded.get(key)
        if k is None:
            k = _loaded[key] = LoadedKernel(built, device_index, sms)
        return k


def unload_all() -> None:
    """Unload every module this process loaded (for tests, and to give the memory back).

    A launch of a kernel may still be running on any stream when this is called (nothing in a launch waits for the device), and a module must not be
    unloaded under a kernel that runs from it: so every context that has a loaded module is synchronized first (``cuCtxSynchronize``, with the context
    made current for the call), and a failure to synchronize (a stream of it is being captured, the context is broken) raises before anything is
    unloaded or forgotten. Nothing is imported or synchronized when no module is loaded.
    """
    with _lock:
        loaded = list(_loaded.items())
    if not loaded:
        return
    drv = _driver()
    contexts: "dict[int, Any]" = {}
    for _, k in loaded:
        contexts.setdefault(int(k.context), k.context)
    for ctx in contexts.values():
        _check(drv.cuCtxPushCurrent(ctx), "cuCtxPushCurrent")
        try:
            _check(drv.cuCtxSynchronize(), "cuCtxSynchronize")  # every launch from these modules, on every stream of the context, has finished
        finally:
            drv.cuCtxPopCurrent()
    with _lock:
        for key, k in loaded:
            if _loaded.get(key) is k:
                del _loaded[key]
            _check(drv.cuCtxPushCurrent(k.context), "cuCtxPushCurrent")  # a module is unloaded from its own context
            try:
                drv.cuModuleUnload(k.module)
            finally:
                drv.cuCtxPopCurrent()


# -- the events that order the runs of a launch plan on one arena (tilefish._planrun) ---------------------------------------------------


def new_event() -> Any:
    """An event of the current context, without timing (``cuEventCreate(CU_EVENT_DISABLE_TIMING)``)."""
    drv = _driver()
    (event,) = _check(drv.cuEventCreate(drv.CUevent_flags.CU_EVENT_DISABLE_TIMING), "cuEventCreate")
    return event


def new_timing_event() -> Any:
    """An event of the current context that records a time (``cuEventCreate(CU_EVENT_DEFAULT)``): for ``tilefish.profile_plan`` only."""
    drv = _driver()
    (event,) = _check(drv.cuEventCreate(drv.CUevent_flags.CU_EVENT_DEFAULT), "cuEventCreate")
    return event


def clock_khz(device_index: int) -> int:
    """The SM clock of a device in kHz (``CU_DEVICE_ATTRIBUTE_CLOCK_RATE``): how many cycles a spinning kernel waits per millisecond."""
    drv = _driver()
    (device,) = _check(drv.cuDeviceGet(device_index), "cuDeviceGet")
    (rate,) = _check(drv.cuDeviceGetAttribute(drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device), "cuDeviceGetAttribute(CLOCK_RATE)")
    return int(rate)


def elapsed_ms(start: Any, end: Any) -> float:
    """The time between the most recent records of two timing events, in milliseconds (``cuEventElapsedTime``; both done)."""
    (ms,) = _check(_driver().cuEventElapsedTime(start, end), "cuEventElapsedTime")
    return float(ms)


def wait_event(stream: int, event: Any, external: bool) -> None:
    """Make the work submitted to ``stream`` after this wait for the most recent record of ``event``. ``external`` (while ``stream`` is being
    captured): the wait is captured as an event wait node, so every replay of the graph waits for the most recent record made before it is launched."""
    drv = _driver()
    flags = drv.CUevent_wait_flags.CU_EVENT_WAIT_EXTERNAL if external else drv.CUevent_wait_flags.CU_EVENT_WAIT_DEFAULT
    _check(drv.cuStreamWaitEvent(drv.CUstream(stream), event, flags), "cuStreamWaitEvent")


def record_event(stream: int, event: Any, external: bool) -> None:
    """Record ``event`` on ``stream``. ``external`` (while ``stream`` is being captured): captured as an event record node, so every replay of the
    graph records the event when its work is done (a record without the flag would only join the capture)."""
    drv = _driver()
    if external:
        _check(drv.cuEventRecordWithFlags(event, drv.CUstream(stream), drv.CUevent_record_flags.CU_EVENT_RECORD_EXTERNAL), "cuEventRecordWithFlags")
    else:
        _check(drv.cuEventRecord(event, drv.CUstream(stream)), "cuEventRecord")


def synchronize_event(event: Any) -> None:
    """Wait on the host until the work of the most recent record of ``event`` is done."""
    _check(_driver().cuEventSynchronize(event), "cuEventSynchronize")


def destroy_event(event: Any) -> None:
    _check(_driver().cuEventDestroy(event), "cuEventDestroy")


def copy_async(dst: int, src: int, nbytes: int, stream: int) -> None:
    """Copy ``nbytes`` bytes of device memory from ``src`` to ``dst`` on ``stream`` (``cuMemcpyDtoDAsync``: stream-ordered, and a memcpy node of the
    graph when the stream is being captured). The copy step of a launch plan."""
    drv = _driver()
    _check(drv.cuMemcpyDtoDAsync(drv.CUdeviceptr(dst), drv.CUdeviceptr(src), nbytes, drv.CUstream(stream)), "cuMemcpyDtoDAsync")


def memset_async(dst: int, nbytes: int, stream: int) -> None:
    """Set ``nbytes`` bytes of device memory at ``dst`` to zero on ``stream`` (``cuMemsetD8Async``: stream-ordered, and a memset node of the graph when
    the stream is being captured). The memset step of a launch plan."""
    drv = _driver()
    _check(drv.cuMemsetD8Async(drv.CUdeviceptr(dst), 0, nbytes, drv.CUstream(stream)), "cuMemsetD8Async")


def synchronize_stream(stream: int) -> None:
    """Wait on the host until the work enqueued on ``stream`` is done (``cuStreamSynchronize``)."""
    drv = _driver()
    _check(drv.cuStreamSynchronize(drv.CUstream(stream)), "cuStreamSynchronize")


def capture_id(stream: int) -> "int | None":
    """The id of the capture ``stream`` takes part in (``cuStreamGetCaptureInfo``: unique among the captures of the process), or ``None`` when the stream
    is not capturing. (cuda-python returns the status and the id first, then the graph and the dependencies, whose number differs between versions.)"""
    drv = _driver()
    status, ident = _check(drv.cuStreamGetCaptureInfo(drv.CUstream(stream)), "cuStreamGetCaptureInfo")[:2]
    if status != drv.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
        return None
    return int(ident)
