"""tilefish's own on-disk cache of builds, keyed by tilefish's own digests and verified on every read.

``build`` compiles from scratch (``always_compile``): Triton's disk cache is keyed by ``JITFunction.cache_key``, which
does not see a ``tl.constexpr`` read as a module attribute, a dtype alias or a constant in a helper module, so it could
serve the cubin of an older kernel (see :mod:`tilefish._source`). That costs about 35 to 130 ms per binary. This cache
gives repeated builds of unchanged sources their speed back without trusting Triton's key.

Where. ``$TILEFISH_CACHE_DIR`` when that is set (an empty value switches the cache off), else ``~/.cache/tilefish``. The
entries are ``<dir>/builds/<key[:2]>/<key>.json``. Nothing evicts them; delete the directory to start again.

The key is a sha256 over

* the declaration digest and the fingerprint of the compile spec (everything that goes into ``triton.compile``: the
  signature, the constants, the attributes, the grid and block, the launch parameters);
* the identity of the source: ``source_sha256`` of a generated kernel (the sha256 of the Triton source its spec was made
  from) or the *source digest* of a hand-written one (:func:`tilefish._source.source_digest`: the files it reaches and the
  Triton version);
* the architecture, the Triton version, the ``ptxas`` version, the environment variables that Triton itself says
  invalidate its cache, and the compile options in effect (the spec's and any a caller added).

``build`` checks that the process holds the files it digests (an edited module that was not reloaded, a reassigned
constant) *before* it computes the key, so a stale process cannot be served an entry for a source it does not run, nor
store one under the digest of a source it did not compile. A kernel whose source digest cannot be computed is not cached.

The value is one JSON file: the cubin (base64), the PTX, and what ``build`` returns besides them. A read recomputes
everything it can instead of believing the file: the ``sha256`` of the cubin, its ``sha256_code`` (the machine code, by
parsing the ELF file), the sha256 of the PTX, the entry point and its parameters against the spec (the ABI check of a
fresh build), the options and the versions against what is being built. Anything that is not as it was written is a miss
with a :class:`~tilefish.errors.TilefishCacheWarning` and the kernel is built again (and the entry rewritten); the cache
never hands back a binary it could not check, and it never fails a build.

A write is a temporary file in the directory and an atomic rename, like every file tilefish writes, so a reader sees the old
entry or the new one and processes that build the same kernel at the same time do not tear it. A symbolic link where an
entry goes is refused.

What this is not: a defence against someone who can write the cache directory. An entry that was rewritten *consistently*
(a cubin with its digests) passes the checks, as an entry of Triton's own cache would; a manifest is the thing to pin
(``Manifest.digest``), and ``build(..., cache=False)`` never reads or writes the cache. It is also only as exact as the source
digest: a dependency that digest does not reach (a function held in a container, an import inside a function body; see
:mod:`tilefish._source`) is not part of the key either. And the toolchain is its version strings: a locally patched Triton that
keeps its version string is not told apart (Triton's own ``triton_key`` hashes its sources and ``libtriton.so``, 220 ms on a
B300, once per process, which would cost more than a hit saves; a development build carries its commit in the version).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import pathlib
import warnings
from typing import Any, Mapping

from ._build import BuiltKernel, LaunchAttributes, _check_abi, write_file
from ._elf import code_digest, parse_elf
from .errors import TilefishBuildError, TilefishCacheWarning

CACHE_FORMAT = "tilefish.build-cache.v1"
ENV_DIR = "TILEFISH_CACHE_DIR"
_TAG = b"tilefish.build-cache-key.v1\0"
_warned_writes: "set[str]" = set()


def cache_directory() -> "pathlib.Path | None":
    """The directory of the cache: ``$TILEFISH_CACHE_DIR``, else ``~/.cache/tilefish``; ``None`` when it is switched off or unknown."""
    env = os.environ.get(ENV_DIR)
    if env is not None:
        return pathlib.Path(env).expanduser() if env.strip() else None
    try:
        return pathlib.Path.home() / ".cache" / "tilefish"
    except (RuntimeError, KeyError):
        return None


def triton_environment() -> "dict[str, str]":
    """The environment variables that Triton says invalidate its own cache, as it reads them (empty where it cannot say)."""
    try:
        from triton._C.libtriton import get_cache_invalidating_env_vars

        return {str(k): str(v) for k, v in dict(get_cache_invalidating_env_vars()).items()}
    except Exception:  # noqa: BLE001 - not Triton, or a Triton that does not have it: no extra key
        return {}


def cache_key(
    spec: Any,
    *,
    arch: int,
    source: "tuple[str, str]",
    triton_version: str,
    ptxas_version: str,
    options: "Mapping[str, Any]",
    environment: "Mapping[str, str]",
) -> str:
    """The key of a build (see the module docstring): ``source`` is ``("source_sha256" | "source_digest", value)``."""
    doc = {
        "declaration_digest": spec.declaration_digest,
        "spec": spec._fingerprint(),
        "source": list(source),
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
        "stem": built.spec.stem,
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
    source_digest: "str | None",
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
    for name, value in (("arch", arch), ("triton_version", triton_version), ("ptxas_version", ptxas_version), ("stem", spec.stem)):
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
        source_digest=source_digest,
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
        source_digest: "str | None",
    ) -> "BuiltKernel | None":
        """The cached build for ``key``, or ``None``: a miss, or an entry that failed its checks (a :class:`~tilefish.errors.TilefishCacheWarning` says which)."""
        path = self.path_of(key)
        try:
            data = path.read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            return None
        except OSError as e:
            warnings.warn(f"cannot read the tilefish build cache entry {str(path)!r} ({e}); building instead", TilefishCacheWarning, stacklevel=3)
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
                source_digest=source_digest,
            )
        except _Invalid as e:
            warnings.warn(
                f"the tilefish build cache entry {str(path)!r} is corrupt or stale ({e}); it is not used, and the kernel is built again",
                TilefishCacheWarning,
                stacklevel=3,
            )
            return None

    def store(self, key: str, built: BuiltKernel) -> bool:
        """Write the entry of ``built`` (atomically); ``False`` (with one warning per directory) when the directory cannot be written."""
        path = self.path_of(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_file(path, encode(key, built))
            return True
        except (OSError, TilefishBuildError) as e:
            if str(self.directory) not in _warned_writes:
                _warned_writes.add(str(self.directory))
                warnings.warn(
                    f"cannot write the tilefish build cache in {str(self.directory)!r} ({e}); builds are not cached there "
                    f"(set {ENV_DIR} to a writable directory, or to an empty value to switch the cache off)",
                    TilefishCacheWarning,
                    stacklevel=3,
                )
            return False


def default_cache() -> "BuildCache | None":
    """The cache at :func:`cache_directory`, or ``None`` when it is switched off."""
    directory = cache_directory()
    return None if directory is None else BuildCache(directory)
