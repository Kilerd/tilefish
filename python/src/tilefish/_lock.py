"""Locked exports: an export that needs no GPU, from a lock file made once on a GPU (``docs/en/guide/locked-export.md``).

An export learns some facts from the device it runs on: the cuBLASLt algorithm and the cuDNN engine each library entry pins (and the cuBLASLt and
cuDNN versions that chose them, and the SM count of the device), the verification of each ``cublaslt.matmul`` pin, the canary digests, and the
verification of a graph's fused regions. Everything else (every cubin: Triton compiles ahead of time for an architecture, ptxas assembles PTX) is a
function of the sources, the declarations and the toolchain.

* :func:`write_lock` reads an export directory (made on a GPU of the target device) and writes its **lock**: every device fact the directory holds,
  the toolchain versions it records, and the sha256 of every file. It reads files only.
* ``tilefish.runtime.export(ops, directory, lock=path)`` is the **locked export**: it builds every binary for the lock's architecture, takes every
  device fact from the lock and never asks a heuristic, runs nothing on a device (with or without one), writes the files an export writes, and
  holds every file to the lock's sha256 (:func:`verify_tree`). A fact the lock does not have refuses the export with what to do: re-lock on a GPU.

The lock is canonical JSON (sorted keys, one space of indent), so a re-lock that changes nothing changes no byte of it, and a review of its diff
reads what changed: a pin, a version, a file. Its format is ``tilefish.lock.v1`` (the guide page lists every field).
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

from .errors import TilefishError, TilefishLockError, TilefishManifestError

if TYPE_CHECKING:
    from ._build import BuiltKernel

LOCK_SCHEMA = "tilefish.lock.v1"
TOP_KEYS = ("schema", "tilefish", "device", "toolchain", "library", "fusion", "kernels", "digests", "files")
TOOLCHAIN_KEYS = {"triton": str, "ptxas": str, "cublaslt": int, "cudnn": int}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX128 = re.compile(r"[0-9a-f]{128}\Z")
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_ARCH = re.compile(r"sm_([0-9]{2,3})\Z")
_STEM = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
CONV_IMPL = "cudnn.conv_fwd"
RELOCK = "re-lock on a GPU of the target device (export there, then tilefish.runtime.write_lock)"
RELOCK_HERE = "re-lock with this toolchain on a GPU of the target device (export there, then tilefish.runtime.write_lock)"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(doc: "dict[str, Any]") -> bytes:
    """The bytes of a lock document as :func:`write_lock` writes it."""
    return (json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


# -- the files of an export directory ---------------------------------------------------------------------------------------------------------


def tree(directory: pathlib.Path) -> "dict[str, str]":
    """The sha256 of every regular file under ``directory``, by its path relative to it (``/``-separated), except ``trace/`` (the trace of a traced
    export is outside the pinned artefact, as it is outside every digest). A symbolic link is refused: an export writes none."""
    from ._trace import TRACE_DIR

    def unreadable(e: OSError) -> NoReturn:
        raise TilefishLockError(f"cannot read the export directory {str(directory)!r}: {e}")

    out: "dict[str, str]" = {}
    for root, dirs, files in os.walk(directory, onerror=unreadable):
        here = pathlib.Path(root)
        rel = here.relative_to(directory)
        for name in [*dirs, *files]:
            if (here / name).is_symlink():
                raise TilefishLockError(f"{str(here / name)!r} is a symbolic link; an export directory has none")
        if rel == pathlib.Path("."):
            dirs[:] = [d for d in dirs if d != TRACE_DIR]
        for name in files:
            key = (rel / name).as_posix() if rel != pathlib.Path(".") else name
            out[key] = _sha256((here / name).read_bytes())
    return dict(sorted(out.items()))


# -- making a lock -----------------------------------------------------------------------------------------------------------------------------


def _one(values: "dict[str, set[Any]]", key: str, value: Any, what: str) -> None:
    values.setdefault(key, set()).add(value)
    if len(values[key]) > 1:
        raise TilefishLockError(
            f"the export directory has binaries of more than one {what} ({sorted(map(str, values[key]))}): a lock is of one export, made with one "
            "toolchain on one device; export into an empty directory and lock that"
        )


def _library_record(entry: "dict[str, Any]") -> "dict[str, Any]":
    """What a lock keeps of a library entry: what the device chose and measured, and the declaration digest it was chosen for."""
    rec: "dict[str, Any]" = {"impl": entry["impl"], "declaration_digest": entry["declaration_digest"], "workspace_bytes": entry["workspace_bytes"]}
    if entry["impl"] == CONV_IMPL:
        rec["pin"] = entry["pin"]
    else:
        rec["algo"] = entry["algo"]
        if "contract" in entry:  # a cublaslt.matmul: the verification of its pin
            rec["contract"] = entry["contract"]
            rec["verification"] = entry["verification"]
    if "canary" in entry:
        rec["canary"] = entry["canary"]
    return rec


def lock_of(directory: "str | os.PathLike[str]") -> "dict[str, Any]":
    """The lock document of an export directory (:func:`write_lock`). Reads files only: the manifest (verified as a load verifies it: every cubin
    against its digests), the plans (each against its digest and the manifest's), the fusion reports, and the sha256 of every file."""
    from . import __version__
    from ._build import CUTE_DIGEST_KIND
    from ._fusion import REPORT_SCHEMA
    from ._manifest import MANIFEST_NAME, Manifest
    from ._plan import PLAN_NAME, PLAN_SET_NAME, plan_file_name, read_plan, read_plan_set

    directory = pathlib.Path(directory)
    if not (directory / MANIFEST_NAME).is_file():
        raise TilefishLockError(f"{str(directory)!r} has no {MANIFEST_NAME}: a lock is made from the directory tilefish.runtime.export wrote")
    m = Manifest.load(directory, instrumented=True)
    if m.doc.get("tilefish") != __version__:
        raise TilefishLockError(
            f"the manifest in {str(directory)!r} was written by tilefish {m.doc.get('tilefish')!r}, and this is tilefish {__version__}: a locked export "
            "writes this version's files, so lock an export of this version"
        )
    arches = sorted({e["arch"] for e in [*m.doc["kernels"], *m.doc["library"]]})
    if len(arches) != 1:
        raise TilefishLockError(
            f"the manifest in {str(directory)!r} has entries for {arches or 'no architecture'}: a lock is of one device, so lock a directory of one "
            "architecture's export"
        )
    arch = arches[0]
    seen: "dict[str, set[Any]]" = {}
    toolchain: "dict[str, Any]" = {}
    kernels: "dict[str, Any]" = {}
    for e in m.doc["kernels"]:
        if e.get("source_digest_kind") == CUTE_DIGEST_KIND:
            raise TilefishLockError(
                f"'{e['stem']}' is a CuTe DSL kernel: its build compiles it on a GPU and captures its launch there, which a locked export cannot do "
                "without one; a directory with a CuTe DSL kernel is not locked (docs/en/guide/locked-export.md, limitations)"
            )
        for key in ("triton", "ptxas"):
            if e["toolchain"].get(key):
                _one(seen, key, e["toolchain"][key], f"{key} version")
                toolchain[key] = e["toolchain"][key]
        kernels[e["stem"]] = {"cubin": e["cubin"], "sha256": e["sha256"], "sha256_code": e["sha256_code"]}
    library: "dict[str, Any]" = {}
    sm_counts: "dict[str, set[Any]]" = {}
    for e in m.doc["library"]:
        version_key = "cudnn" if e["impl"] == CONV_IMPL else "cublaslt"
        version = e["cudnn_version" if e["impl"] == CONV_IMPL else "cublaslt_version"]
        _one(seen, version_key, version, f"{'cuDNN' if version_key == 'cudnn' else 'cuBLASLt'} version")
        toolchain[version_key] = version
        _one(sm_counts, "sm_count", e["sm_count"], "SM count (the library entries were pinned on devices of other SM counts)")
        library[e["stem"]] = _library_record(e)
    sm_count = next(iter(sm_counts["sm_count"])) if sm_counts else None
    known = {MANIFEST_NAME, *(k["cubin"] for k in kernels.values())}
    digests: "dict[str, Any]" = {"manifest": m.digest, "plans": {}}
    docs = []
    if (directory / PLAN_SET_NAME).is_file():
        index = read_plan_set(directory)
        if index["manifest_digest"] != m.digest:
            raise TilefishLockError(f"{PLAN_SET_NAME} in {str(directory)!r} is of another manifest than the one next to it: lock the directory an export wrote")
        for name, digest in index["plans"].items():
            doc = read_plan(directory / plan_file_name(name))
            if doc["plan_digest"] != digest or doc["manifest_digest"] != m.digest:
                raise TilefishLockError(f"the plan of '{name}' in {str(directory)!r} is not the one {PLAN_SET_NAME} lists: lock the directory an export wrote")
            digests["plans"][name] = digest
            known.add(plan_file_name(name))
            docs.append(doc)
        digests["set"] = index["set_digest"]
        known.add(PLAN_SET_NAME)
    elif (directory / PLAN_NAME).is_file():
        doc = read_plan(directory)
        if doc["manifest_digest"] != m.digest:
            raise TilefishLockError(f"{PLAN_NAME} in {str(directory)!r} is of another manifest than the one next to it: lock the directory an export wrote")
        digests["plans"][doc["graph"]] = doc["plan_digest"]
        known.add(PLAN_NAME)
        docs.append(doc)
    if docs:  # an export of graphs writes the entries its plans launch, and no other (one left from an export of another graph is not this export's)
        launched = {st["entry"] for d in docs for b in d["buckets"] for st in b["steps"] if "entry" in st}
        stray = sorted({*kernels, *library} - launched)
        if stray:
            raise TilefishLockError(
                f"the manifest in {str(directory)!r} has entries no plan of the directory launches ({', '.join(stray[:5])}{' ...' if len(stray) > 5 else ''}): "
                "they are left from another export into it; a lock is of one export, so export into an empty directory and lock that"
            )
    fusion: "dict[str, Any]" = {}
    for report_file in sorted(directory.glob("*.fusion.json")) if digests["plans"] else []:  # <graph>.fusion.json, of a graph whose body computes
        known.add(report_file.name)
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise TilefishLockError(f"cannot read the fusion report {str(report_file)!r}: {e}") from None
        name = report.get("graph") if isinstance(report, dict) else None
        regions = report.get("regions") if isinstance(report, dict) else None
        if (
            not isinstance(report, dict)
            or report.get("schema") != REPORT_SCHEMA
            or f"{name}.fusion.json" != report_file.name
            or not isinstance(regions, list)
            or any(not isinstance(item, dict) or not isinstance(item.get("op"), str) or not isinstance(item.get("verified", {}), dict) for item in regions)
        ):
            raise TilefishLockError(f"{str(report_file)!r} is not the fusion report of a graph of the export")
        if name not in digests["plans"]:
            raise TilefishLockError(f"{str(report_file)!r} is the fusion report of '{name}', which has no plan in {str(directory)!r}: it is left from another export")
        verified = []
        for i, item in enumerate(regions):
            if "verified" not in item:
                continue
            if "skipped" in item["verified"]:
                raise TilefishLockError(
                    f"the fused regions of '{name}' were not verified by the export ({item['verified']['skipped']}): a lock records what a device "
                    "verified, so export on a GPU of the lock's architecture (or with verify=False) and lock that"
                )
            verified.append({"region": i, "op": item["op"], "verified": item["verified"]})
        if verified:
            fusion[name] = verified
    files = tree(directory)
    unknown = sorted(set(files) - known)
    if unknown:
        raise TilefishLockError(
            f"{str(directory)!r} has files an export does not write ({', '.join(unknown[:5])}{' ...' if len(unknown) > 5 else ''}): a lock holds every file "
            "of the directory, so lock a directory that holds one export and nothing else"
        )
    return {
        "schema": LOCK_SCHEMA,
        "tilefish": __version__,
        "device": {"arch": arch, "sm_count": sm_count},
        "toolchain": dict(sorted(toolchain.items())),
        "library": dict(sorted(library.items())),
        "fusion": dict(sorted(fusion.items())),
        "kernels": dict(sorted(kernels.items())),
        "digests": digests,
        "files": files,
    }


def write_lock(directory: "str | os.PathLike[str]", path: "str | os.PathLike[str]") -> "dict[str, Any]":
    """Write the lock of the export ``directory`` to ``path`` (canonical JSON: :func:`lock_of`) and return it. The file is written through a
    temporary file and renamed, and read back (:func:`read_lock`)."""
    from ._build import write_file

    doc = lock_of(directory)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_file(path, canonical(doc))
    read_lock(path)
    return doc


# -- reading a lock ----------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Lock:
    """A lock file, read and checked (:func:`read_lock`)."""

    path: str
    doc: "dict[str, Any]"

    @property
    def arch(self) -> str:
        return self.doc["device"]["arch"]

    @property
    def cap(self) -> int:
        return int(self.arch[3:])

    @property
    def sm_count(self) -> "int | None":
        return self.doc["device"]["sm_count"]

    @property
    def toolchain(self) -> "dict[str, Any]":
        return self.doc["toolchain"]

    @property
    def device_words(self) -> str:
        return f"{self.arch}" + (f" with {self.sm_count} SMs" if self.sm_count is not None else "")


def _bad(path: str, message: str) -> NoReturn:
    raise TilefishLockError(f"the lock {path!r} is malformed: {message}")


def _keys(path: str, where: str, value: Any, required: "tuple[str, ...]", optional: "tuple[str, ...]" = ()) -> None:
    if not isinstance(value, dict):
        _bad(path, f"{where} must be an object, got {type(value).__name__}")
    missing = [k for k in required if k not in value]
    unknown = sorted(str(k) for k in set(value) - set(required) - set(optional))
    if missing or unknown:
        _bad(path, f"{where} must have the fields {list(required)}" + (f" (and may have {list(optional)})" if optional else "")
             + (f"; missing {missing}" if missing else "") + (f"; unknown {unknown}" if unknown else ""))  # fmt: skip


def _int(value: Any, low: int) -> bool:
    return type(value) is int and value >= low


def _check_canary(path: str, where: str, value: Any, digest_key: str) -> None:
    _keys(path, f"{where}.canary", value, ("seed", digest_key))
    if not (_int(value["seed"], 0) and value["seed"] < 1 << 32 and isinstance(value[digest_key], str) and _HEX32.match(value[digest_key])):
        _bad(path, f"{where}.canary must be {{\"seed\": an int from 0 to 2**32 - 1, \"{digest_key}\": 32 lowercase hex digits}}, got {value!r}")


def _check_library(path: str, stem: str, rec: Any) -> None:
    from . import _cudnn
    from ._library import CONTRACTS, CUBLASLT_IMPLS, IMPL_MATMUL, VERIFICATION_KEYS

    where = f"library.{stem}"
    impl = rec.get("impl") if isinstance(rec, dict) else None
    if impl == CONV_IMPL:
        _keys(path, where, rec, ("impl", "declaration_digest", "workspace_bytes", "pin"), ("canary",))
        try:
            _cudnn.pin_of(rec["pin"], f"{where}.pin")
        except TilefishError as e:
            _bad(path, str(e))
        if "canary" in rec:
            _check_canary(path, where, rec["canary"], "y_digest")
    elif impl in CUBLASLT_IMPLS:
        matmul = impl == IMPL_MATMUL
        required = ("impl", "declaration_digest", "workspace_bytes", "algo") + (("contract", "verification") if matmul else ())
        _keys(path, where, rec, required, ("canary",) if matmul else ())
        if not (isinstance(rec["algo"], str) and _HEX128.match(rec["algo"])):
            _bad(path, f"{where}.algo must be 128 lowercase hex digits (the 64 bytes of a cublasLtMatmulAlgo_t)")
        if matmul:
            if rec["contract"] not in CONTRACTS:
                _bad(path, f"{where}.contract must be one of {list(CONTRACTS)}, got {rec['contract']!r}")
            _keys(path, f"{where}.verification", rec["verification"], tuple(VERIFICATION_KEYS))
            if "canary" in rec:
                _check_canary(path, where, rec["canary"], "out_digest")
    else:
        _bad(path, f"{where}.impl must be one of {sorted([*CUBLASLT_IMPLS, CONV_IMPL])}, got {impl!r}")
    if not (isinstance(rec["declaration_digest"], str) and _HEX64.match(rec["declaration_digest"])):
        _bad(path, f"{where}.declaration_digest must be 64 lowercase hex digits")
    if not (_int(rec["workspace_bytes"], 1) and rec["workspace_bytes"] <= 1 << 40):
        _bad(path, f"{where}.workspace_bytes must be an int from 1 to 2**40, got {rec['workspace_bytes']!r}")


def _check_doc(path: str, doc: Any) -> None:
    _keys(path, "the document", doc, TOP_KEYS)
    if doc["schema"] != LOCK_SCHEMA:
        _bad(path, f"schema must be {LOCK_SCHEMA!r}, got {doc['schema']!r}")
    if not isinstance(doc["tilefish"], str):
        _bad(path, "tilefish must be a version string")
    _keys(path, "device", doc["device"], ("arch", "sm_count"))
    if not (isinstance(doc["device"]["arch"], str) and _ARCH.match(doc["device"]["arch"])):
        _bad(path, f"device.arch must look like 'sm_103', got {doc['device']['arch']!r}")
    if doc["device"]["sm_count"] is not None and not _int(doc["device"]["sm_count"], 1):
        _bad(path, f"device.sm_count must be a positive int or null, got {doc['device']['sm_count']!r}")
    _keys(path, "toolchain", doc["toolchain"], (), tuple(TOOLCHAIN_KEYS))
    for key, value in doc["toolchain"].items():
        if (TOOLCHAIN_KEYS[key] is int and not _int(value, 1)) or (TOOLCHAIN_KEYS[key] is str and not (isinstance(value, str) and value)):
            _bad(path, f"toolchain.{key} must be a {'positive int' if TOOLCHAIN_KEYS[key] is int else 'non-empty string'}, got {value!r}")
    for section in ("library", "fusion", "kernels", "files"):
        if not isinstance(doc[section], dict):
            _bad(path, f"{section} must be an object")
    for stem, rec in doc["library"].items():
        if not _STEM.match(stem):
            _bad(path, f"library has a key that is not a stem: {stem!r}")
        _check_library(path, stem, rec)
        version_key = "cudnn" if rec["impl"] == CONV_IMPL else "cublaslt"
        if version_key not in doc["toolchain"] or doc["device"]["sm_count"] is None:
            _bad(path, f"library.{stem} is a {rec['impl']} pin, so toolchain.{version_key} and device.sm_count must be there")
    for graph, regions in doc["fusion"].items():
        if not isinstance(regions, list) or not regions:
            _bad(path, f"fusion.{graph} must be a non-empty list")
        for k, r in enumerate(regions):
            _keys(path, f"fusion.{graph}[{k}]", r, ("region", "op", "verified"))
            if not _int(r["region"], 0) or not isinstance(r["op"], str) or not isinstance(r["verified"], dict):
                _bad(path, f"fusion.{graph}[{k}] must be {{\"region\": an index, \"op\": a name, \"verified\": an object}}")
    for stem, k in doc["kernels"].items():
        if not _STEM.match(stem):
            _bad(path, f"kernels has a key that is not a stem: {stem!r}")
        _keys(path, f"kernels.{stem}", k, ("cubin", "sha256", "sha256_code"))
        if not (isinstance(k["cubin"], str) and "/" not in k["cubin"] and k["cubin"] not in ("", ".", "..")):
            _bad(path, f"kernels.{stem}.cubin must be a plain file name, got {k['cubin']!r}")
        for key in ("sha256", "sha256_code"):
            if not (isinstance(k[key], str) and _HEX64.match(k[key])):
                _bad(path, f"kernels.{stem}.{key} must be 64 lowercase hex digits")
    _keys(path, "digests", doc["digests"], ("manifest", "plans"), ("set",))
    for key, value in [("manifest", doc["digests"]["manifest"]), ("set", doc["digests"].get("set", "0" * 64))]:
        if not (isinstance(value, str) and _HEX64.match(value)):
            _bad(path, f"digests.{key} must be 64 lowercase hex digits")
    if not isinstance(doc["digests"]["plans"], dict) or any(not (isinstance(v, str) and _HEX64.match(v)) for v in doc["digests"]["plans"].values()):
        _bad(path, "digests.plans must map graph names to 64 lowercase hex digits")
    for name, digest in doc["files"].items():
        parts = name.split("/")
        if not name or name.startswith("/") or any(p in ("", ".", "..") for p in parts) or parts[0] == "trace":
            _bad(path, f"files has a path that is not a relative path inside the directory: {name!r}")
        if not (isinstance(digest, str) and _HEX64.match(digest)):
            _bad(path, f"files.{name} must be 64 lowercase hex digits")


def read_lock(path: "str | os.PathLike[str]") -> Lock:
    """Read and check a lock file (strict JSON: no duplicate key, no ``NaN``). Raises :class:`~tilefish.errors.TilefishLockError`."""
    from ._manifest import _no_duplicate_keys, _parse_float, _reject_constant

    shown = str(path)
    try:
        text = pathlib.Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise TilefishLockError(f"cannot read the lock {shown!r}: {e}") from None
    try:
        doc = json.loads(text, parse_constant=_reject_constant, parse_float=_parse_float, object_pairs_hook=_no_duplicate_keys)
    except (ValueError, RecursionError, TilefishManifestError) as e:
        raise TilefishLockError(f"the lock {shown!r} is not valid JSON: {e}") from None
    _check_doc(shown, doc)
    return Lock(shown, doc)


def lock_arg(lock: Any) -> Lock:
    """``export(..., lock=)``: a path to a lock file (or a :class:`Lock`)."""
    if isinstance(lock, Lock):
        return lock
    if isinstance(lock, (str, os.PathLike)):
        return read_lock(lock)
    raise TilefishLockError(f"lock= is the path of a lock file (tilefish.runtime.write_lock wrote it), got {lock!r}")


# -- the locked export ---------------------------------------------------------------------------------------------------------------------------


def target(lock: Lock, arch: "int | str | None") -> int:
    """The architecture of a locked export: the lock's; an ``arch=`` that is another is refused (the lock is of another device). Also checks the
    tilefish version and the toolchain that compiles (:func:`check_versions`)."""
    from ._build import parse_arch

    if arch is not None and parse_arch(arch) != lock.cap:
        raise TilefishLockError(
            f"the lock {lock.path!r} is of a device of {lock.device_words}, and this export is for sm_{parse_arch(arch)}: a lock is of one device; use "
            f"the lock of the sm_{parse_arch(arch)} device (or make one: export on a GPU of it and tilefish.runtime.write_lock)"
        )
    check_versions(lock)
    return lock.cap


def check_versions(lock: Lock) -> None:
    """The tilefish version and the toolchain versions the lock was made with must be this process's (another one writes other bytes)."""
    from . import __version__

    if lock.doc["tilefish"] != __version__:
        raise TilefishLockError(
            f"the lock {lock.path!r} was made by tilefish {lock.doc['tilefish']}, and this is tilefish {__version__}: the manifest and the plans name the "
            f"version that writes them, so no file would be the lock's; install tilefish {lock.doc['tilefish']}, or {RELOCK_HERE}"
        )
    toolchain = lock.toolchain
    if "triton" in toolchain:
        try:
            import triton
        except ImportError:
            raise TilefishLockError(f"the lock {lock.path!r} has binaries Triton {toolchain['triton']} built, and Triton is not installed here") from None
        if str(triton.__version__) != toolchain["triton"]:
            raise TilefishLockError(
                f"the lock {lock.path!r} was made with Triton {toolchain['triton']}, and this process has Triton {triton.__version__}: another Triton "
                f"compiles other binaries; install Triton {toolchain['triton']}, or {RELOCK_HERE}"
            )
    if "ptxas" in toolchain:
        from . import _binary, _build

        try:
            here = _build.ptxas_version() if "triton" in toolchain else _build._ptxas_version(_binary.find_ptxas())
        except TilefishError as e:
            raise TilefishLockError(f"the lock {lock.path!r} has binaries assembled by ptxas ({toolchain['ptxas']}), and none is found here: {e}") from None
        if here != toolchain["ptxas"]:
            raise TilefishLockError(
                f"the lock {lock.path!r} was made with the ptxas '{toolchain['ptxas']}', and this process compiles with '{here}': another ptxas "
                f"assembles other cubins; set TRITON_PTXAS_PATH (or TILEFISH_PTXAS_PATH) to the lock's ptxas, or {RELOCK_HERE}"
            )


def check_built(built: "BuiltKernel", lock: Lock) -> None:
    """A binary the locked export built must be the lock's (its toolchain, its sha256), before anything is written."""
    rec = lock.doc["kernels"].get(built.stem)
    if rec is None:
        raise TilefishLockError(
            f"the export builds '{built.stem}' ({built.arch}), and the lock {lock.path!r} has no binary of that name: the lock is of another set of "
            f"operators, or this one is new; {RELOCK}"
        )
    for key, here in (("triton", built.triton_version), ("ptxas", built.ptxas_version)):
        if here and lock.toolchain.get(key) != here:
            raise TilefishLockError(
                f"'{built.stem}' ({built.arch}) was built with the {key} '{here}', and the lock {lock.path!r} was made with "
                f"'{lock.toolchain.get(key)}': install the lock's, or {RELOCK_HERE}"
            )
    if built.sha256 != rec["sha256"]:
        if built.sha256_code == rec["sha256_code"]:
            why = "its machine code (sha256_code) is the lock's, so its other sections differ (the debug information names the source files: is TILEFISH_DEBUG_PATHS set?)"
        else:
            why = (
                f"its machine code differs too (sha256_code {built.sha256_code} here, {rec['sha256_code']} in the lock): its source, its declaration "
                "or the toolchain is not the one the lock was made from"
            )
        raise TilefishLockError(
            f"the cubin of '{built.stem}' ({built.arch}) is not the lock's: sha256 {built.sha256} here, {rec['sha256']} in the lock; {why}. If the change "
            f"is meant, {RELOCK}"
        )


def refuse_cute(stem: str, lock: Lock) -> NoReturn:
    raise TilefishLockError(
        f"'{stem}' is a CuTe DSL kernel, which is compiled on a GPU and whose launch is captured there: a locked export runs without a GPU and cannot "
        "build it (docs/en/guide/locked-export.md, limitations); export it on a GPU"
    )


def locked_library(spec: Any, lock: Lock, cap: int) -> Any:
    """The pinned library entry of ``spec`` (a ``LibrarySpec`` or a ``ConvSpec``) from the lock, for ``sm_<cap>``: what the GPU export pinned, measured
    and recorded, with no device, no library and no heuristic. The lock's pin must be of this declaration on the lock's device (its
    ``declaration_digest``); a stem the lock does not have is refused."""
    from . import _conv, _cudnn
    from ._library import CANARY_KEYS, VERIFICATION_KEYS, BuiltLibrary

    arch = f"sm_{cap}"
    rec = lock.doc["library"].get(spec.stem)
    impl = spec.impl
    if rec is None:
        raise TilefishLockError(
            f"the lock {lock.path!r} has no pin for '{spec.stem}' ({impl}) on {arch}: the lock was made without this operator (or before it was "
            f"declared), and a locked export never asks a heuristic; {RELOCK}"
        )
    if rec["impl"] != impl:
        raise TilefishLockError(f"the lock {lock.path!r} pins '{spec.stem}' as a {rec['impl']}, and the declaration is a {impl}; {RELOCK}")
    sms = lock.sm_count
    assert sms is not None  # (read_lock: a library record needs the device's SM count)
    want = spec.pinned_digest(sms)
    if rec["declaration_digest"] != want or rec["workspace_bytes"] != spec.workspace_bytes:
        raise TilefishLockError(
            f"the lock {lock.path!r} pins '{spec.stem}' ({impl}) for another declaration: its declaration digest is {rec['declaration_digest']}, and this "
            f"declaration's on a device of {sms} SMs is {want} (the operator, its options or its buckets changed since the lock was made); {RELOCK}"
        )
    if isinstance(spec, _conv.ConvSpec):
        pin = _cudnn.pin_of(rec["pin"], f"the lock's pin of '{spec.stem}'")
        if pin.api != spec.api:
            raise TilefishLockError(f"the lock {lock.path!r} has a {pin.api} pin for '{spec.stem}', and the declaration's api is {spec.api!r}; {RELOCK}")
        if "canary" not in rec:  # (an export records a canary in every convolution entry)
            raise TilefishLockError(f"the lock {lock.path!r} has no canary for the cuDNN convolution '{spec.stem}'; {RELOCK}")
        return _valid(_conv.BuiltConv(spec, arch, pin, lock.toolchain["cudnn"], sms, _ordered(rec["canary"], ("seed", "y_digest"))), lock)
    version = lock.toolchain["cublaslt"]
    if not spec.matmul:
        return _valid(BuiltLibrary(spec, arch, rec["algo"], version, sms), lock)
    if spec.contract == "measured-exact" and rec["contract"] != "measured-exact":
        raise TilefishLockError(
            f"the lock {lock.path!r} records the pin of '{spec.stem}' as {rec['contract']}, and the declaration asks for contract='measured-exact' (an "
            f"export refuses such a pin); {RELOCK}"
        )
    if spec.canary and "canary" not in rec:
        raise TilefishLockError(f"the lock {lock.path!r} has no canary for '{spec.stem}', and its declaration has one (canary=True); {RELOCK}")
    canary = _ordered(rec["canary"], CANARY_KEYS) if spec.canary else None
    return _valid(BuiltLibrary(spec, arch, rec["algo"], version, sms, rec["contract"], _ordered(rec["verification"], VERIFICATION_KEYS), canary), lock)


def _valid(built: Any, lock: Lock) -> Any:
    """``built`` (from the lock), after its entry passed every rule a manifest reader holds a library entry to (``validate_entry``): a record of the
    lock that would make an entry no host reads is refused here, before anything is written."""
    from ._library import validate_entry

    try:
        validate_entry(json.loads(json.dumps(built.entry())))
    except TilefishManifestError as e:
        raise TilefishLockError(f"the lock {lock.path!r} has a record of '{built.stem}' that makes no valid manifest entry: {e}") from None
    return built


def _ordered(record: "dict[str, Any]", keys: "tuple[str, ...]") -> "dict[str, Any]":
    """A record of the lock (whose keys are sorted) in the order of the keys the export writes it in: the manifest is written in the order of its
    dicts, not sorted."""
    return {k: record[k] for k in keys}


def locked_fusion(graph: Any, verify: "bool | None", lock: Lock) -> "dict[int, Any] | None":
    """What the export's verification of ``graph``'s fused regions found, from the lock (the ``checked`` of :func:`tilefish._fusion.verify_for_export`,
    which the fusion report records): ``None`` when the export verifies nothing. A region the export would verify that the lock has no record of is
    refused: re-lock on a GPU."""
    from . import _fusion
    from .errors import TilefishRuntimeError

    if verify is not None and not isinstance(verify, bool):
        raise TilefishRuntimeError(f"verify= is True, False or None (the graph's tilefish.fusion(verify=...)), got {verify!r}")
    if not _fusion.has_arithmetic(graph):
        return None
    want = _fusion.settings_of(graph).verify if verify is None else verify
    f = _fusion.fused(graph)
    if not want or not (f.fused_regions or f.hooked):
        return None
    recorded = {r["region"]: r for r in lock.doc["fusion"].get(f.source.name, [])}
    position = {id(r): i for i, r in enumerate(f.regions)}
    out: "dict[int, Any]" = {}
    for r in [*f.fused_regions, *(h.region for h in f.hooked)]:
        i = position[id(r)]
        h = f.hooked_region(r)
        op = h.op.name if h is not None else r.kernel.name
        rec = recorded.get(i)
        if rec is None or rec["op"] != op:
            raise TilefishLockError(
                f"the lock {lock.path!r} has no verification of the fused region {i} ('{op}') of graph '{graph.name}': a locked export runs nothing on a "
                f"device, and the region is verified on one; {RELOCK} (or export with verify=False)"
            )
        out[r.index] = rec["verified"]
    return out


def verify_tree(directory: "str | os.PathLike[str]", lock: Lock) -> None:
    """Every file of ``directory`` (but ``trace/``) must be a file of the lock, with its sha256, and every file of the lock must be there. Raises
    :class:`~tilefish.errors.TilefishLockError` naming the first difference: a missing file, then a file that differs (the cubins first, then the
    manifest, then the plans and the rest), then a file the lock does not have."""
    from ._manifest import MANIFEST_NAME

    directory = pathlib.Path(directory)
    have = tree(directory)
    want = lock.doc["files"]
    missing = [name for name in want if name not in have]
    if missing:
        raise TilefishLockError(
            f"the export did not write {missing[0]!r}, which the lock {lock.path!r} has: the lock is of an export of other operators (or of other "
            f"options: buckets=, reuse=, verify=); {RELOCK}"
        )
    cubins = {k["cubin"] for k in lock.doc["kernels"].values()}

    def order(name: str) -> "tuple[int, str]":
        return (0 if name in cubins else 1 if name == MANIFEST_NAME else 2, name)

    for name in sorted(want, key=order):
        if have[name] != want[name]:
            what = "the manifest" if name == MANIFEST_NAME else "a cubin" if name in cubins else "a file"
            raise TilefishLockError(
                f"{what} the export wrote, {name!r}, is not the lock's: sha256 {have[name]} here, {want[name]} in the lock {lock.path!r}. A locked "
                f"export writes what an export on a GPU writes with the lock's pins; if the change is meant, {RELOCK}"
            )
    extra = sorted(set(have) - set(want))
    if extra:
        raise TilefishLockError(
            f"{str(directory)!r} has {extra[0]!r}, a file the lock {lock.path!r} does not have: export into an empty directory (a locked export holds "
            f"every file of it to the lock), or {RELOCK}"
        )
    # the directory locks to the very lock: what the files say (the digests to pin, the binaries, the pins, the regions) is what the lock says, so a lock
    # whose files and digests disagree (a digest a host would be told to pin, edited by hand) is refused too
    from .aot import _first_difference, _show

    found = _first_difference(lock.doc, lock_of(directory), "")
    if found is not None:
        path, want, got = found
        raise TilefishLockError(
            f"the lock {lock.path!r} is not the lock of the files it holds: at {path} it has {_show(want)}, and the directory, whose every file is the "
            f"lock's, gives {_show(got)}; the lock was edited by hand: {RELOCK}"
        )


__all__ = ["LOCK_SCHEMA", "Lock", "lock_of", "read_lock", "verify_tree", "write_lock"]
