"""CUDA C and PTX references for the gpu tests (not collected: the name has no ``test_`` prefix). Needs torch and cuda-python.

A reference is compiled by NVRTC from CUDA C, or is PTX, and launched through the CUDA driver on torch's current stream, so the tests
compare a lowered kernel with what CUDA C++ compiles the same function to, on the same GPU. NVRTC is loaded with ``ctypes`` by path,
so any release can be the compiler: by default the one torch ships (``nvidia.cuda_nvrtc``, the CUDA of torch's build), or the
``libnvrtc.so`` that ``$TILEFISH_NVRTC`` names. NVRTC's options are nvcc's defaults (``-ftz=false -prec-div=true -prec-sqrt=true
-fmad=true``) unless a test says otherwise; the PTX is for the highest virtual architecture NVRTC supports up to the device's, and
the driver compiles it for the device (the driver's own ``ptxas``; the numerics of the PTX instructions do not depend on it).
"""

from __future__ import annotations

import ctypes
import functools
import glob
import os

import torch
from cuda.bindings import driver as drv


class ReferenceUnavailable(RuntimeError):
    """A reference cannot be built here (no NVRTC, or a driver too old for the PTX): the test skips with this message."""


def _check(result, what):
    err = result[0] if isinstance(result, tuple) else result
    if err != drv.CUresult.CUDA_SUCCESS:
        name = getattr(err, "name", str(err))
        raise RuntimeError(f"{what} failed: {name}")
    return tuple(result[1:]) if isinstance(result, tuple) else ()


def _torch_nvrtc() -> "str | None":
    try:
        import nvidia.cuda_nvrtc as pkg  # torch's own NVRTC (CUDA 12 wheels)

        roots = list(pkg.__path__)
    except ImportError:
        roots = []
    try:
        import nvidia.cu13 as pkg13  # CUDA 13 wheels put every library under nvidia/cu13

        roots += list(pkg13.__path__)
    except ImportError:
        pass
    for root in roots:
        found = sorted(glob.glob(os.path.join(root, "lib", "libnvrtc.so.*[0-9]")))
        found = [f for f in found if "builtins" not in f and ".alt." not in f]
        if found:
            return found[-1]
    return None


class Nvrtc:
    """One NVRTC library, loaded by path."""

    def __init__(self, path: str) -> None:
        self.path = path
        # NVRTC opens its builtins library by name when it compiles: load the one next to it first, so the name resolves to it
        for builtins in sorted(glob.glob(os.path.join(os.path.dirname(path), "libnvrtc-builtins.so.*"))):
            if ".alt." not in builtins:
                ctypes.CDLL(builtins, mode=ctypes.RTLD_GLOBAL)
        self.lib = ctypes.CDLL(path)
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._call("nvrtcVersion", ctypes.byref(major), ctypes.byref(minor))
        self.version = (major.value, minor.value)
        count = ctypes.c_int()
        self._call("nvrtcGetNumSupportedArchs", ctypes.byref(count))
        archs = (ctypes.c_int * count.value)()
        self._call("nvrtcGetSupportedArchs", archs)
        self.archs = tuple(archs)

    def _call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status != 0:
            self.lib.nvrtcGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(f"{name}: {self.lib.nvrtcGetErrorString(status).decode()}")

    def arch_for(self, capability: int) -> int:
        """The highest architecture this NVRTC supports that the device runs PTX of (at most the device's)."""
        usable = [a for a in self.archs if a <= capability]
        if not usable:
            raise ReferenceUnavailable(f"NVRTC {self.version} supports none of the architectures up to sm_{capability}")
        return max(usable)

    def ptx(self, source: str, options=(), name: str = "reference.cu") -> str:
        prog = ctypes.c_void_p()
        self._call("nvrtcCreateProgram", ctypes.byref(prog), source.encode(), name.encode(), 0, None, None)
        try:
            opts = [o.encode() for o in options]
            array = (ctypes.c_char_p * len(opts))(*opts)
            status = self.lib.nvrtcCompileProgram(prog, len(opts), array)
            if status != 0:
                size = ctypes.c_size_t()
                self.lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
                log = ctypes.create_string_buffer(size.value)
                self.lib.nvrtcGetProgramLog(prog, log)
                raise RuntimeError(f"NVRTC {self.version} could not compile the reference:\n{log.value.decode(errors='replace')}")
            size = ctypes.c_size_t()
            self._call("nvrtcGetPTXSize", prog, ctypes.byref(size))
            buf = ctypes.create_string_buffer(size.value)
            self._call("nvrtcGetPTX", prog, buf)
            return buf.value.decode()
        finally:
            self.lib.nvrtcDestroyProgram(ctypes.byref(prog))


@functools.lru_cache(maxsize=None)
def nvrtc(path: "str | None" = None) -> Nvrtc:
    """The NVRTC of ``path``, else ``$TILEFISH_NVRTC``, else torch's own; :class:`ReferenceUnavailable` when there is none."""
    path = path or os.environ.get("TILEFISH_NVRTC") or _torch_nvrtc()
    if not path:
        raise ReferenceUnavailable("no NVRTC library: install torch's nvidia-cuda-nvrtc wheel or set TILEFISH_NVRTC to a libnvrtc.so")
    return Nvrtc(path)


def capability() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


NVCC_DEFAULTS = ("-ftz=false", "-prec-div=true", "-prec-sqrt=true", "-fmad=true")


def include_dirs() -> "list[str]":
    """The CUDA headers of the runtime wheels (``cuda_fp16.h``, ``cuda_bf16.h``), which NVRTC does not have built in."""
    out = []
    for pkg in ("nvidia.cuda_runtime", "nvidia.cu13"):
        try:
            module = __import__(pkg, fromlist=["__path__"])
        except ImportError:
            continue
        out += [os.path.join(root, "include") for root in module.__path__ if os.path.isdir(os.path.join(root, "include"))]
    return out


def compile_cuda(source: str, options=NVCC_DEFAULTS, compiler: "Nvrtc | None" = None) -> str:
    """The PTX NVRTC makes of ``source`` (extern "C" kernels) with ``options`` (nvcc's defaults), for the device."""
    c = compiler or nvrtc()
    if "#include" in source:
        dirs = include_dirs()
        if not dirs:
            raise ReferenceUnavailable("the reference includes CUDA headers, and no CUDA runtime wheel (nvidia-cuda-runtime) has them here")
        options = (*options, *(f"-I{d}" for d in dirs))
    return c.ptx(source, [f"--gpu-architecture=compute_{c.arch_for(capability())}", *options])


class Module:
    """A PTX (or cubin) image loaded into torch's context; :meth:`launch` runs one of its kernels on torch's current stream."""

    def __init__(self, image: "str | bytes") -> None:
        torch.zeros(1, device="cuda")  # the primary context is current
        data = image.encode() + b"\0" if isinstance(image, str) else image
        try:
            (self.module,) = _check(drv.cuModuleLoadData(data), "cuModuleLoadData")
        except RuntimeError as e:
            if "UNSUPPORTED_PTX_VERSION" in str(e):
                raise ReferenceUnavailable(f"the driver cannot compile this PTX (its ISA is newer than the driver's): {e}") from e
            raise
        self._functions = {}

    def function(self, name: str):
        if name not in self._functions:
            (self._functions[name],) = _check(drv.cuModuleGetFunction(self.module, name.encode()), f"cuModuleGetFunction({name})")
        return self._functions[name]

    def launch(self, name: str, grid, block, *args, shared: int = 0) -> None:
        """Launch ``name``: a tensor is passed as its device pointer, a python int as an ``int``, a python float as a ``float``, a
        ctypes value as it is."""
        values = []
        for a in args:
            if isinstance(a, torch.Tensor):
                values.append(ctypes.c_uint64(a.data_ptr()))
            elif isinstance(a, bool) or not isinstance(a, (int, float, ctypes._SimpleCData)):
                raise TypeError(f"cannot pass {a!r} to a kernel")
            elif isinstance(a, int):
                values.append(ctypes.c_int32(a))
            elif isinstance(a, float):
                values.append(ctypes.c_float(a))
            else:
                values.append(a)
        slots = (ctypes.c_void_p * len(values))(*[ctypes.addressof(v) for v in values])
        g = tuple(grid) + (1,) * (3 - len(grid))
        b = tuple(block) + (1,) * (3 - len(block))
        stream = drv.CUstream(torch.cuda.current_stream().cuda_stream)
        _check(drv.cuLaunchKernel(self.function(name), *g, *b, shared, stream, ctypes.addressof(slots), 0), f"cuLaunchKernel({name})")


@functools.lru_cache(maxsize=None)
def cuda_module(source: str, options=NVCC_DEFAULTS) -> Module:
    """:func:`compile_cuda` of ``source``, loaded (one module per source and options in a process)."""
    return Module(compile_cuda(source, options))


ELEMENTWISE_BLOCK = 256


def map_f32(module: Module, name: str, *inputs: torch.Tensor) -> torch.Tensor:
    """``out[i] = f(inputs[0][i], ...)`` by the kernel ``name(const T* in0, ..., T* out, int n)`` of ``module``: one thread per element."""
    n = inputs[0].numel()
    out = torch.empty_like(inputs[0])
    module.launch(name, ((n + ELEMENTWISE_BLOCK - 1) // ELEMENTWISE_BLOCK,), (ELEMENTWISE_BLOCK,), *inputs, out, n)
    return out
