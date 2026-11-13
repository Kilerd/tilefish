"""tilefish's own on-disk cache of builds, keyed by the Triton IR of the kernel and verified on every read.

``build`` compiles from scratch (``always_compile``): Triton's disk cache is keyed by ``JITFunction.cache_key``, which does not see
a ``tl.constexpr`` read as a module attribute, through ``getattr``, from a class, a ``triton.constexpr_function`` that reads the
environment, and so on, so it could serve the cubin of another kernel. That costs 25 to 75 ms per binary on a B300. This cache gives
repeated builds their speed back without trusting Triton's key, and without guessing what Python state Triton reads.

Where. ``$TILEFISH_CACHE_DIR`` when that is set (an empty value switches the cache off), else ``~/.cache/tilefish``. The
entries are ``<dir>/builds/<key[:2]>/<key>.json``. Nothing evicts them; delete the directory to start again.

The key is a sha256 over

* the cache recipe (:data:`CACHE_RECIPE`, a number that is bumped whenever tilefish changes what goes into a binary) and the version
  of tilefish;
* the **IR digest** (:mod:`tilefish._ir`): the TTIR that Triton's frontend builds for the kernel in this process, without locations
  or names. It stands for the kernel: the body, every constant the kernel reads by any route, the jit functions it calls. It is the
  only thing that says which kernel it is; nothing is looked for in source files, namespaces or closures;
* what the stages after the IR take besides it: the compile options in effect (the spec's and any a caller added), the target
  architecture, the Triton version, the ``ptxas`` version, and the Triton settings that change code generation
  (:func:`codegen_environment`, listed below).

``codegen_environment`` is, for Triton 3.5.1: every environment variable whose name starts with ``TRITON_``, ``MLIR_``, ``LLVM_``,
``NVPTX_``, ``AMDGCN_``, ``DISABLE_``, ``ENABLE_`` or ``ALLOW_``, or is ``USE_IR_LOC``, except those that are known not to change the
code (:data:`NOT_CODEGEN`: cache and dump directories, the tools' paths whose version is in the key, the launcher's paths,
debugging and autotuning switches, remote cache settings). That is a superset of the 30 variables that Triton's own
``get_cache_invalidating_env_vars()`` lists (``DISABLE_LLVM_OPT``, ``DISABLE_MMA_V3``, ``DISABLE_MMA_V5``, ``DISABLE_PTXAS_OPT``,
``TRITON_DEFAULT_FP_FUSION``, ``TRITON_F32_DEFAULT``, ``TRITON_DISABLE_LINE_INFO``, ``TRITON_ENABLE_ASAN``, ``TRITON_OVERRIDE_ARCH``,
``USE_IR_LOC``, ``LLVM_PASS_PLUGIN_PATH``, the dump and timing switches, ...), and it has the ones that list does not:
``TRITON_LIBDEVICE_PATH``, ``TRITON_MOCK_PTX_VERSION``, ``TRITON_KERNEL_OVERRIDE`` (a build refuses it), ``TRITON_FRONT_END_DEBUGGING``,
``TRITON_ALLOW_NON_CONSTEXPR_GLOBALS``, ``TRITON_ENABLE_EXPERIMENTAL_CONSAN``. A variable that a later Triton adds is in the key
unless :data:`NOT_CODEGEN` names it: a missing entry in that list can only cost a miss. Next to the variables, the *effective* value
of each codegen knob as ``triton.knobs`` reports it (a program may set a knob without the environment): ``compilation`` (``override``,
``use_ir_loc``, ``enable_asan``, ``disable_line_info``, ``front_end_debugging``, ``allow_non_constexpr_globals``,
``enable_experimental_consan``), ``language`` (``fp32_default``, ``default_fp_fusion``), ``nvidia`` (``disable_ptxas_opt``,
``mock_ptx_version``, ``libdevice_path``), and the sha256 of the file that ``libdevice_path`` names.

A kernel that Triton's frontend cannot build has no digest, so it has no key (``build`` raises). The IR is computed *before* the cache
is looked up, so the lookup costs a run of the frontend (2 to 4 ms), and a binary is only ever stored under the digest of the IR it
was compiled from (``build`` compiles from a source that records the IR it was given, and refuses a mismatch).

The value is one JSON file: the cubin (base64), the PTX, the IR digest, and what ``build`` returns besides them. A read recomputes
everything it can instead of believing the file: the ``sha256`` of the cubin, its ``sha256_code`` (the machine code, by
parsing the ELF file), the sha256 of the PTX, the entry point and its parameters against the spec (the ABI check of a
fresh build), the options, the IR digest and the versions against what is being built. Anything that is not as it was written is a
miss with a :class:`~tilefish.errors.TilefishCacheWarning` and the kernel is built again (and the entry rewritten); the cache
never hands back a binary it could not check, and it never fails a build.

Never fails a build, also when warnings are errors. The warning is a :class:`~tilefish.errors.TilefishCacheWarning`, and under
``python -W error`` (or ``pytest -W error``, ``warnings.simplefilter("error")``) ``warnings.warn`` raises it, from inside ``build``,
after the kernel compiled fine. The cache therefore issues its warnings through a helper that catches what the filters and the hooks
raise and logs the message instead (``logging.getLogger("tilefish.cache")``, WARNING): with ``-W error`` the text is in the log and
the build succeeds. The same holds for anything unexpected that reading or writing an entry raises: a miss, with a warning.

A write is a temporary file in the directory and an atomic rename, like every file tilefish writes, so a reader sees the old
entry or the new one and processes that build the same kernel at the same time do not tear it. A symbolic link where an
entry goes is refused.

What this is not: a defence against someone who can write the cache directory. An entry that was rewritten *consistently*
(a cubin with its digests) passes the checks, as an entry of Triton's own cache would; a manifest is the thing to pin
(``Manifest.digest``), and ``build(..., cache=False)`` never reads or writes the cache. The toolchain is its version strings: a
locally patched Triton that keeps its version string is not told apart (Triton's own ``triton_key`` hashes its sources and
``libtriton.so``, 220 ms on a B300, once per process, which would cost more than a hit saves; a development build carries its
commit in the version).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import pathlib
import warnings
from typing import Any, Mapping

from ._build import BuiltKernel, LaunchAttributes, _check_abi, write_file
from ._elf import code_digest, parse_elf
from .errors import TilefishBuildError, TilefishCacheWarning

CACHE_FORMAT = "tilefish.build-cache.v2"
# The recipe: what tilefish does to turn a compile spec into a binary (how a spec becomes the inputs of ``triton.compile``, the
# options it adds, the ABI it checks, how a generated kernel's source is written, how the IR digest is made). It is part of every
# key, so a change of any of that must bump it: an entry made by an older recipe is then not found, instead of served as the binary
# of the new one. The tilefish version is part of the key as well (a release can change what goes into a binary without anyone
# remembering to bump this).
CACHE_RECIPE = 3  # 3: a binary is keyed by the IR digest of its kernel (the source digest of 2 is gone), and by the Triton knobs that change code generation
ENV_DIR = "TILEFISH_CACHE_DIR"
_TAG = b"tilefish.build-cache-key.v1\0"
_warned_writes: "set[str]" = set()
_log = logging.getLogger("tilefish.cache")


def _warn(message: str) -> None:
    """Tell the user something about the cache, in a way that cannot fail the build that is running.

    The cache is an optimisation: a build that compiled fine must not fail because the cache did not work. A
    :class:`~tilefish.errors.TilefishCacheWarning` is a warning, but ``python -W error`` (or ``warnings.simplefilter("error")``, or
    ``pytest -W error``) makes ``warnings.warn`` *raise* it, from inside ``build``, and so would a ``showwarning`` hook that fails.
    So the warning is issued where the filters can only turn it into an exception that is caught here, and the message is then logged
    (``logging.getLogger("tilefish.cache")``, at WARNING) so that it is not lost: the user who asked for warnings to be errors
    sees the same text in the log, and the build goes on.
    """
    try:
        warnings.warn(message, TilefishCacheWarning, stacklevel=4)
    except Exception:  # noqa: BLE001 - an "error" filter raises the warning itself, a hook may raise anything: neither fails a build
        try:
            _log.warning("%s", message)
        except Exception:  # noqa: BLE001 - nor does a logging handler that fails
            pass


def cache_directory() -> "pathlib.Path | None":
    """The directory of the cache: ``$TILEFISH_CACHE_DIR``, else ``~/.cache/tilefish``; ``None`` when it is switched off or unknown."""
    env = os.environ.get(ENV_DIR)
    if env is not None:
        return pathlib.Path(env).expanduser() if env.strip() else None
    try:
        return pathlib.Path.home() / ".cache" / "tilefish"
    except (RuntimeError, KeyError):
        return None


_ENV_PREFIXES = ("TRITON_", "MLIR_", "LLVM_", "NVPTX_", "AMDGCN_", "DISABLE_", "ENABLE_", "ALLOW_")
# Environment variables of those families that do not change the code Triton generates: where its caches and dumps go, the
# tools whose *version* is in the key (``ptxas``) or that only disassemble (``cuobjdump``, ``nvdisasm``), the paths the launcher
# stub is built from (tilefish never builds one), debugging, autotuning and remote cache switches. A variable that is not named
# here is part of the key.
NOT_CODEGEN = frozenset(
    {
        "TRITON_CACHE_DIR",
        "TRITON_HOME",
        "TRITON_DUMP_DIR",
        "TRITON_OVERRIDE_DIR",
        "TRITON_CACHE_MANAGER",
        "TRITON_REMOTE_CACHE_BACKEND",
        "TRITON_REDIS_KEY_FORMAT",
        "TRITON_REDIS_HOST",
        "TRITON_REDIS_PORT",
        "TRITON_PTXAS_PATH",
        "TRITON_CUOBJDUMP_PATH",
        "TRITON_NVDISASM_PATH",
        "TRITON_CUDACRT_PATH",
        "TRITON_CUDART_PATH",
        "TRITON_LIBCUDA_PATH",
        "TRITON_LIBHIP_PATH",
        "TRITON_CUPTI_LIB_PATH",
        "TRITON_INTERPRET",
        "TRITON_DEBUG",
        "TRITON_PRINT_AUTOTUNING",
        "TRITON_CACHE_AUTOTUNING",
        "TRITON_KERNEL_DUMP",
        "TRITON_STORE_BINARY_ONLY",
        "TRITON_ALWAYS_COMPILE",
        "TRITON_DUMP_PTXAS_LOG",
        "TRITON_PROFILE",
        "TRITON_BUILD_WITH_CCACHE",
        "TRITON_BUILD_WITH_CLANG_LLD",
    }
)
_KNOBS = {
    "compilation": (
        "override",
        "use_ir_loc",
        "enable_asan",
        "disable_line_info",
        "front_end_debugging",
        "allow_non_constexpr_globals",
        "enable_experimental_consan",
    ),
    "language": ("fp32_default", "default_fp_fusion"),
    "nvidia": ("disable_ptxas_opt", "mock_ptx_version", "libdevice_path"),
}


def codegen_environment() -> "dict[str, Any]":
    """What Triton's later stages read from the environment and its knobs, as it is now (see the module docstring for the list).

    ``env`` is the environment variables of the codegen families, ``knobs`` the effective value of each codegen knob as
    ``triton.knobs`` reports it (a knob can be set by a program without the environment), and ``libdevice`` the sha256 of the
    libdevice file when ``TRITON_LIBDEVICE_PATH`` (or the knob) names one. Empty where Triton is not importable.
    """
    out: dict[str, Any] = {
        "env": {
            k: v
            for k, v in sorted(os.environ.items())
            if (k.startswith(_ENV_PREFIXES) or k == "USE_IR_LOC") and k not in NOT_CODEGEN
        }
    }
    try:
        import triton
    except ImportError:
        return out
    knobs: dict[str, str] = {}
    libdevice_path = None
    for group, names in _KNOBS.items():
        holder = getattr(getattr(triton, "knobs", None), group, None)
        for name in names:
            try:
                value = getattr(holder, name)
                knobs[f"{group}.{name}"] = repr(value)
                if (group, name) == ("nvidia", "libdevice_path"):
                    libdevice_path = value
            except Exception as e:  # noqa: BLE001 - a knob this Triton does not have, or one that raises: said, so that it is in the key
                knobs[f"{group}.{name}"] = f"<{type(e).__name__}>"
    out["knobs"] = knobs
    if libdevice_path:
        try:
            with open(libdevice_path, "rb") as f:
                out["libdevice"] = hashlib.sha256(f.read()).hexdigest()
        except Exception as e:  # noqa: BLE001
            out["libdevice"] = f"<unreadable: {type(e).__name__}>"
    return out


def cache_key(
    *,
    arch: int,
    ir_digest: str,
    triton_version: str,
    ptxas_version: str,
    options: "Mapping[str, Any]",
    environment: "Mapping[str, Any]",
) -> str:
    """The key of a build (see the module docstring): the IR digest and what the stages after the IR take besides it."""
    from . import __version__

    doc = {
        "recipe": CACHE_RECIPE,
        "tilefish": __version__,
        "ir": ir_digest,
        "arch": int(arch),
        "triton": triton_version,
        "ptxas": ptxas_version,
        "environment": dict(environment),
        "options": dict(options),
    }
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False, default=repr)
    return hashlib.sha256(_TAG + text.encode("utf-8")).hexdigest()


def encode(key: str, built: BuiltKernel) -> bytes:
    """The bytes of the entry of ``built`` under ``key``."""
    doc = {
        "format": CACHE_FORMAT,
        "key": key,
        "ir_digest": built.ir_digest,
        "arch": built.arch,
        "symbol": built.symbol,
        "shared": built.shared,
        "num_warps": built.num_warps,
        "sha256": built.sha256,
        "sha256_code": built.sha256_code,
        "ptx_sha256": hashlib.sha256(built.ptx.encode("utf-8")).hexdigest(),
        "triton_version": built.triton_version,
        "ptxas_version": built.ptxas_version,
        "ptx_target": built.ptx_target,
        "options": dict(built.options),
        "launch": built.launch.to_dict(),
        "ptx": built.ptx,
        "cubin": base64.b64encode(built.cubin).decode("ascii"),
    }
    return (json.dumps(doc, sort_keys=True, allow_nan=False, default=repr) + "\n").encode("utf-8")


class _Invalid(Exception):
    """An entry that is not what was written (the message says how)."""


def _field(doc: "dict[str, Any]", name: str, kind: type) -> Any:
    value = doc.get(name)
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise _Invalid(f"the field '{name}' is missing or not a {kind.__name__}")
    return value


def decode(
    data: bytes,
    key: str,
    spec: Any,
    *,
    arch: str,
    options: "Mapping[str, Any]",
    triton_version: str,
    ptxas_version: str,
    ir_digest: str,
) -> BuiltKernel:
    """The :class:`~tilefish.aot.BuiltKernel` an entry holds, after checking it; raises :class:`_Invalid` for anything that is not as it was written."""
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise _Invalid(f"it is not JSON ({type(e).__name__})") from e
    if not isinstance(doc, dict):
        raise _Invalid("it is not a JSON object")
    if doc.get("format") != CACHE_FORMAT:
        raise _Invalid(f"its format is {doc.get('format')!r}, not {CACHE_FORMAT!r}")
    if doc.get("key") != key:
        raise _Invalid("it is the entry of another key")
    try:
        cubin = base64.b64decode(_field(doc, "cubin", str), validate=True)
    except (binascii.Error, ValueError) as e:
        raise _Invalid("its cubin is not base64") from e
    ptx = _field(doc, "ptx", str)
    symbol = _field(doc, "symbol", str)
    sha256, sha256_code = _field(doc, "sha256", str), _field(doc, "sha256_code", str)
    if hashlib.sha256(cubin).hexdigest() != sha256:
        raise _Invalid("the sha256 of its cubin is not the recorded one")
    if hashlib.sha256(ptx.encode("utf-8")).hexdigest() != _field(doc, "ptx_sha256", str):
        raise _Invalid("the sha256 of its PTX is not the recorded one")
    try:
        elf = parse_elf(cubin)
        code = code_digest(cubin)
    except TilefishBuildError as e:
        raise _Invalid(f"its cubin is not a valid ELF file: {e}") from e
    if code != sha256_code:
        raise _Invalid("the sha256_code (machine code) of its cubin is not the recorded one")
    if elf.section(f".text.{symbol}") is None:
        raise _Invalid(f"its cubin has no code section '.text.{symbol}'")
    wanted = json.loads(json.dumps(dict(options), default=repr))  # as JSON reads it back
    if doc.get("options") != wanted:
        raise _Invalid(f"its compile options {doc.get('options')!r} are not {wanted!r}")
    for name, value in (("arch", arch), ("triton_version", triton_version), ("ptxas_version", ptxas_version), ("ir_digest", ir_digest)):
        if doc.get(name) != value:
            raise _Invalid(f"its {name} is {doc.get(name)!r}, not {value!r}")
    num_warps = _field(doc, "num_warps", int)
    if num_warps != wanted.get("num_warps", spec.num_warps):
        raise _Invalid(f"its num_warps is {num_warps}, not what the options ask for")
    try:
        _check_abi(spec, ptx, symbol)
        launch = LaunchAttributes.from_dict(_field(doc, "launch", dict))
    except TilefishBuildError as e:
        raise _Invalid(str(e)) from e
    except (KeyError, TypeError, ValueError) as e:
        raise _Invalid(f"its launch attributes are malformed ({type(e).__name__}: {e})") from e
    return BuiltKernel(
        spec=spec,
        arch=arch,
        cubin=cubin,
        ptx=ptx,
        symbol=symbol,
        shared=_field(doc, "shared", int),
        num_warps=num_warps,
        sha256=sha256,
        sha256_code=sha256_code,
        triton_version=triton_version,
        ptxas_version=ptxas_version,
        ptx_target=_field(doc, "ptx_target", str),
        options=dict(options),
        launch=launch,
        ir_digest=ir_digest,
    )


class BuildCache:
    """The entries of one cache directory."""

    def __init__(self, directory: "str | os.PathLike[str]") -> None:
        self.directory = pathlib.Path(directory)

    def path_of(self, key: str) -> pathlib.Path:
        return self.directory / "builds" / key[:2] / f"{key}.json"

    def load(
        self,
        key: str,
        spec: Any,
        *,
        arch: str,
        options: "Mapping[str, Any]",
        triton_version: str,
        ptxas_version: str,
        ir_digest: str,
    ) -> "BuiltKernel | None":
        """The cached build for ``key``, or ``None``: a miss, or an entry that failed its checks (a :class:`~tilefish.errors.TilefishCacheWarning` says which)."""
        path = self.path_of(key)
        try:
            data = path.read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            return None
        except OSError as e:
            _warn(f"cannot read the tilefish build cache entry {str(path)!r} ({e}); building instead")
            return None
        try:
            return decode(
                data,
                key,
                spec,
                arch=arch,
                options=options,
                triton_version=triton_version,
                ptxas_version=ptxas_version,
                ir_digest=ir_digest,
            )
        except _Invalid as e:
            _warn(f"the tilefish build cache entry {str(path)!r} is corrupt or stale ({e}); it is not used, and the kernel is built again")
            return None
        except Exception as e:  # noqa: BLE001 - whatever a damaged entry makes the checks raise: it is a miss, never a failed build
            _warn(
                f"the tilefish build cache entry {str(path)!r} could not be checked ({type(e).__name__}: {e}); it is not used, "
                "and the kernel is built again"
            )
            return None

    def store(self, key: str, built: BuiltKernel) -> bool:
        """Write the entry of ``built`` (atomically); ``False`` (with one warning per directory) when the directory cannot be written."""
        path = self.path_of(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_file(path, encode(key, built))
            return True
        except Exception as e:  # noqa: BLE001 - a directory that cannot be written, or an entry that cannot be made: the build is done
            if str(self.directory) not in _warned_writes:
                _warned_writes.add(str(self.directory))
                _warn(
                    f"cannot write the tilefish build cache in {str(self.directory)!r} ({type(e).__name__}: {e}); builds are not "
                    f"cached there (set {ENV_DIR} to a writable directory, or to an empty value to switch the cache off)"
                )
            return False


def default_cache() -> "BuildCache | None":
    """The cache at :func:`cache_directory`, or ``None`` when it is switched off."""
    directory = cache_directory()
    return None if directory is None else BuildCache(directory)
