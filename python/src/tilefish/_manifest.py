"""Reading a manifest directory back: the cubins that ``replay`` runs, checked against what the manifest says.

``write_manifest(built, path, cubins=True)`` writes ``manifest.json`` and one ``<stem>.<arch>.cubin`` per
kernel. :meth:`Manifest.load` reads them back and refuses anything that is not what was written: a missing or
malformed file, a cubin whose ``sha256`` (the file) or ``sha256_code`` (the machine code) differs from the
manifest's, a file name that points out of the directory, or an entry that two kernels share.
:meth:`Manifest.built_kernel` then hands a replay the binary for one compile spec, after checking that the
entry is what that spec asks for (its ABI, grid, block, constants, preconditions, options and declaration
digest), so a replay can never run a binary under assumptions it was not compiled for.

Trust. A manifest directory vouches for nothing by itself: its digests only say that the files are the ones the
manifest describes, and an attacker who can write the directory can rewrite both. :attr:`Manifest.digest` is a sha256
over the whole manifest (every entry: spec digests, source digests, file and code digests, symbol, shared memory,
block, launch fields, toolchain); a caller that holds it from a trusted place (a build log, a lock file) passes it as
``pin=`` and a manifest with any other content is refused. Without a pin the directory is trusted.

Source. The compile spec of a generated kernel carries ``source_sha256``, the sha256 of the Triton source it was built
from, and a replay refuses an entry built from another source (a kernel redeclared with another body but the same name
and signature). A hand-written ``extern`` has no source in its declaration, so the entry records a source digest (a sha256
over the files the kernel is made of and the Triton version, see :mod:`tilefish._source`) and a replay recomputes it from
the importable kernel and refuses a mismatch; where it cannot (Triton or the module is not importable), it refuses unless
the caller pinned the manifest, which says that the caller vouches for it.

The file is read as strict JSON: no ``NaN`` or ``Infinity``, no number that does not fit a float, no object with the same
key twice (which parsers disagree about), and every field a replay reads has the type it needs, so a malformed
manifest is a :class:`~tilefish.errors.TilefishManifestError` and never a bare ``KeyError`` or ``AttributeError``.

Nothing here needs Triton, torch or a GPU: it is the part of a host that is just files and digests.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._build import MANIFEST_SCHEMA, BuiltKernel, LaunchAttributes, spec_entry
from ._source import SOURCE_DIGEST_KIND
from ._elf import code_digest, parse_elf, uses_shared
from .errors import TilefishBuildError, TilefishManifestError

if TYPE_CHECKING:
    from .aot import CompileSpec

MANIFEST_NAME = "manifest.json"
DIGEST_TAG = b"tilefish.manifest-digest.v1\0"  # part of what is hashed: another canonical form must change it


def manifest_digest(doc: "dict[str, Any]") -> str:
    """The sha256 (lowercase hex) of a manifest document: its schema and every kernel entry in full.

    Entries are sorted by ``(stem, arch)`` and serialized as canonical JSON (sorted keys, no whitespace), so the
    digest does not depend on the order of the entries or on how the file was formatted, and changes with any field
    of any entry: the spec-derived fields (ABI, grid, block, constants, preconditions, options, declaration digest,
    source digests), the build facts (symbol, shared memory, launch attributes, toolchain, architecture) and the
    digests of the cubin file and of its machine code. The top-level ``tilefish`` version is not part of it.
    """
    try:
        kernels = sorted(doc["kernels"], key=lambda e: (str(e.get("stem")), str(e.get("arch"))))
        text = json.dumps({"schema": doc["schema"], "kernels": kernels}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, AttributeError, KeyError) as e:  # a number that is not finite, a value JSON cannot hold, a malformed entry
        raise TilefishManifestError(f"the manifest cannot be digested, it is not plain JSON data: {type(e).__name__}: {e}") from e
    return hashlib.sha256(DIGEST_TAG + text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, eq=False)
class Artifact:
    """One binary of a manifest: its entry and its cubin, whose digests have been checked."""

    entry: "dict[str, Any]"
    cubin: bytes

    @property
    def stem(self) -> str:
        return self.entry["stem"]

    @property
    def arch(self) -> str:
        return self.entry["arch"]

    @property
    def sha256_code(self) -> str:
        return self.entry["sha256_code"]


def _reject_constant(constant: str) -> Any:
    raise TilefishManifestError(f"the manifest is not strict JSON: it contains the constant {constant}")


def _parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise TilefishManifestError(f"the manifest is not strict JSON: the number {text} does not fit a float")
    return value


def _no_duplicate_keys(pairs: "list[tuple[str, Any]]") -> "dict[str, Any]":
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise TilefishManifestError(f"the manifest is not strict JSON: an object has the key {key!r} twice")
        out[key] = value
    return out


class Manifest:
    """The binaries of a manifest directory, verified when it is loaded."""

    def __init__(
        self, directory: pathlib.Path, doc: "dict[str, Any]", artifacts: "dict[tuple[str, str], Artifact]", path: "pathlib.Path | None" = None
    ) -> None:
        self.directory = directory
        self.path = path if path is not None else directory / MANIFEST_NAME
        self.doc = doc
        self._artifacts = artifacts

    @property
    def digest(self) -> str:
        """The digest of the whole manifest (see :func:`manifest_digest`): what ``run(..., pin=...)`` compares."""
        return manifest_digest(self.doc)

    @classmethod
    def load(cls, path: "str | pathlib.Path") -> "Manifest":
        """Load ``<directory>/manifest.json`` (or the manifest file itself) and every cubin it names.

        Raises :class:`~tilefish.errors.TilefishManifestError` when a file is missing or malformed, a cubin is not
        an ELF file with the code section the manifest's symbol names, or its ``sha256`` or ``sha256_code`` is not
        the one the manifest records.
        """
        path = pathlib.Path(path)
        manifest_path = path / MANIFEST_NAME if path.is_dir() else path
        try:
            text = manifest_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise TilefishManifestError(f"cannot read the manifest {str(manifest_path)!r}: {e}") from e
        try:
            doc = json.loads(text, parse_constant=_reject_constant, parse_float=_parse_float, object_pairs_hook=_no_duplicate_keys)
        except RecursionError as e:
            raise TilefishManifestError(f"the manifest {str(manifest_path)!r} is nested too deeply to be read") from e
        except TilefishManifestError:
            raise
        except ValueError as e:
            raise TilefishManifestError(f"the manifest {str(manifest_path)!r} is not valid JSON: {e}") from e
        if not isinstance(doc, dict) or doc.get("schema") != MANIFEST_SCHEMA or not isinstance(doc.get("kernels"), list):
            raise TilefishManifestError(
                f"the manifest {str(manifest_path)!r} is not a tilefish manifest of schema {MANIFEST_SCHEMA} "
                f"(schema {doc.get('schema') if isinstance(doc, dict) else None!r})"
            )
        artifacts: dict[tuple[str, str], Artifact] = {}
        for i, entry in enumerate(doc["kernels"]):
            artifact = cls._read(manifest_path.parent, i, entry)
            key = (artifact.stem, artifact.arch)
            if key in artifacts:
                raise TilefishManifestError(f"the manifest lists '{key[0]}' for {key[1]} twice")
            artifacts[key] = artifact
        return cls(manifest_path.parent, doc, artifacts, manifest_path)

    @staticmethod
    def _read(directory: pathlib.Path, i: int, entry: Any) -> Artifact:
        if not isinstance(entry, dict):
            raise TilefishManifestError(f"manifest kernel {i} is not an object")
        needed = ("stem", "arch", "symbol", "cubin", "sha256", "sha256_code", "shared", "launch", "options", "toolchain")
        missing = [k for k in needed if k not in entry]
        if missing:
            raise TilefishManifestError(f"manifest kernel {i} ({entry.get('stem')!r}) is missing {missing}")
        stem, arch, name = entry["stem"], entry["arch"], entry["cubin"]
        if not isinstance(name, str) or not name or pathlib.PurePath(name).name != name or name in (".", ".."):
            raise TilefishManifestError(f"'{stem}': the cubin file name {name!r} must be a plain file name in the manifest directory")
        try:
            cubin = (directory / name).read_bytes()
        except OSError as e:
            raise TilefishManifestError(f"'{stem}' ({arch}): cannot read the cubin {name!r}: {e}") from e
        found = hashlib.sha256(cubin).hexdigest()
        if found != entry["sha256"]:
            raise TilefishManifestError(
                f"'{stem}' ({arch}): the file {name!r} has sha256 {found}, but the manifest says {entry['sha256']}; "
                "the cubin is not the one that was built"
            )
        try:
            elf = parse_elf(cubin)
            code = code_digest(cubin)
        except TilefishBuildError as e:
            raise TilefishManifestError(f"'{stem}' ({arch}): {name!r} is not a valid cubin: {e}") from e
        if code != entry["sha256_code"]:
            raise TilefishManifestError(
                f"'{stem}' ({arch}): the machine code of {name!r} has sha256_code {code}, but the manifest says "
                f"{entry['sha256_code']}; the binary was changed after it was built"
            )
        if elf.section(f".text.{entry['symbol']}") is None:
            raise TilefishManifestError(f"'{stem}' ({arch}): {name!r} has no code section '.text.{entry['symbol']}'")
        for field in ("options", "toolchain", "launch"):
            if not isinstance(entry[field], dict):
                raise TilefishManifestError(f"'{stem}' ({arch}): '{field}' must be an object, got {entry[field]!r}")
        shared = entry["shared"]
        if isinstance(shared, bool) or not isinstance(shared, int) or shared < 0:
            raise TilefishManifestError(f"'{stem}' ({arch}): shared must be a non-negative integer, got {shared!r}")
        if shared == 0 and uses_shared(elf, entry["symbol"]):
            raise TilefishManifestError(
                f"'{stem}' ({arch}): the kernel has a shared memory section ('.nv.shared.{entry['symbol']}'), so it uses "
                "shared memory, but the manifest says shared = 0; a launch without the dynamic shared memory it addresses "
                "would fault and poison the CUDA context"
            )
        return Artifact(entry, cubin)

    @property
    def artifacts(self) -> "list[Artifact]":
        return list(self._artifacts.values())

    def __contains__(self, key: "tuple[str, str]") -> bool:
        return key in self._artifacts

    def get(self, stem: str, arch: str) -> "Artifact | None":
        """The artifact of ``stem`` for ``arch`` (``"sm_103"``), or ``None``."""
        return self._artifacts.get((stem, arch))

    def built_kernel(
        self, spec: "CompileSpec", arch: str, *, source_digest: "str | None" = None, pinned: bool = False
    ) -> BuiltKernel:
        """The binary for ``spec`` on ``arch``, as a :class:`~tilefish.aot.BuiltKernel` a replay can launch.

        The manifest entry must be what the spec asks for: every field the spec determines (ABI, grid, block,
        ``num_warps``, constants, buckets, preconditions, compile options, declaration digest and, for a generated
        kernel, the ``source_sha256``) must equal the spec's. For a hand-written kernel (a spec without a
        ``source_sha256``) the entry's ``source_digest`` must equal ``source_digest``, the digest the caller recomputed
        from the importable kernel; when the caller could not (``None``) or the entry has none, the binary is only
        accepted with ``pinned=True``: the caller vouches for the manifest. A recomputed digest that differs from the
        entry's is a refusal either way. The result has no PTX (a manifest does not carry it).
        """
        artifact = self._artifacts.get((spec.stem, arch))
        if artifact is None:
            have = sorted(a for (s, a) in self._artifacts if s == spec.stem)
            raise TilefishManifestError(
                f"the manifest in {str(self.directory)!r} has no binary for '{spec.stem}' on {arch}"
                + (f" (it has {', '.join(have)})" if have else f" (it has {sorted({s for s, _ in self._artifacts})})")
            )
        entry = artifact.entry
        from .aot import _first_difference, _show

        self._check_source(spec, arch, entry, source_digest, pinned)

        expected = json.loads(json.dumps(spec_entry(spec)))  # as JSON read it back: lists, strings, plain numbers
        got = {k: entry.get(k) for k in expected}
        found = _first_difference(expected, got, "")
        if found is not None:
            path, want, have = found
            raise TilefishManifestError(
                f"the manifest entry for '{spec.stem}' ({arch}) is not what the compile spec asks for: first difference "
                f"at {path}: the manifest has {_show(have)}, the spec gives {_show(want)}"
            )
        toolchain = entry["toolchain"]
        try:
            launch = LaunchAttributes.from_dict(entry["launch"])
        except (KeyError, TypeError, ValueError) as e:
            raise TilefishManifestError(f"'{spec.stem}' ({arch}): the launch attributes are malformed: {e}") from e
        try:
            return BuiltKernel(
                spec=spec,
                arch=arch,
                cubin=artifact.cubin,
                ptx="",
                symbol=entry["symbol"],
                shared=int(entry["shared"]),
                num_warps=int(entry["num_warps"]),
                sha256=entry["sha256"],
                sha256_code=entry["sha256_code"],
                triton_version=str(toolchain.get("triton", "")),
                ptxas_version=str(toolchain.get("ptxas", "")),
                ptx_target=str(entry.get("ptx_target", "")),
                options=dict(entry["options"]),
                launch=launch,
                source_digest=entry.get("source_digest"),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise TilefishManifestError(f"'{spec.stem}' ({arch}): the manifest entry is malformed: {type(e).__name__}: {e}") from e

    @staticmethod
    def _check_source(spec: "CompileSpec", arch: str, entry: "dict[str, Any]", source_digest: "str | None", pinned: bool) -> None:
        where = f"'{spec.stem}' ({arch})"
        if spec.source_sha256 is not None:  # a generated kernel: its source is part of its spec
            have = entry.get("source_sha256")
            if have != spec.source_sha256:
                raise TilefishManifestError(
                    f"the manifest entry for {where} was built from kernel source with sha256 {have}, but the kernel now "
                    f"generates source with sha256 {spec.source_sha256}: the kernel's body (or an option of its lowering) changed "
                    "since the manifest was written, so its binary is stale; export it again"
                )
            return
        recorded = entry.get("source_digest")
        kind = entry.get("source_digest_kind")
        if recorded is not None and kind != SOURCE_DIGEST_KIND:
            recorded = None  # a digest of another definition (an older tilefish) says nothing about the files now
            unusable = f"the manifest entry's source digest is of kind {kind!r}, not {SOURCE_DIGEST_KIND!r}"
        else:
            unusable = "the manifest entry records no source digest"
        if source_digest is not None and recorded is not None and source_digest != recorded:
            raise TilefishManifestError(
                f"the manifest entry for {where} was built from kernel files with source digest {recorded}, but the files of "
                f"'{spec.triton}' now have source digest {source_digest}: the kernel, or a module it uses, changed since the "
                "manifest was written, so its binary is stale; export it again (and pin the new manifest)"
            )
        if source_digest is None or recorded is None:
            if pinned:
                return  # the caller vouches for the manifest
            why = (
                unusable
                if recorded is None
                else f"the source digest of '{spec.triton}' cannot be recomputed here (Triton or the module is not importable)"
            )
            raise TilefishManifestError(
                f"cannot tell whether the binary for {where} was built from the current source of the hand-written kernel "
                f"'{spec.triton}': {why}. Pass pin=<the manifest's digest> to vouch for the manifest, or rebuild it"
            )

    def __repr__(self) -> str:
        return f"<Manifest {str(self.directory)!r}: {len(self._artifacts)} binaries>"
