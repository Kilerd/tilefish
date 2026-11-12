"""Reading a manifest directory back: the cubins that ``replay`` runs, checked against what the manifest says.

``write_manifest(built, path, cubins=True)`` writes ``manifest.json`` and one ``<stem>.<arch>.cubin`` per
kernel. :meth:`Manifest.load` reads them back and refuses anything that is not what was written: a missing or
malformed file, a cubin whose ``sha256`` (the file) or ``sha256_code`` (the machine code) differs from the
manifest's, a file name that points out of the directory, or an entry that two kernels share.
:meth:`Manifest.built_kernel` then hands a replay the binary for one compile spec, after checking that the
entry is what that spec asks for (its ABI, grid, block, constants, preconditions, options and declaration
digest), so a replay can never run a binary under assumptions it was not compiled for.

Nothing here needs Triton, torch or a GPU: it is the part of a host that is just files and digests.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._build import MANIFEST_SCHEMA, BuiltKernel, LaunchAttributes, spec_entry
from ._elf import code_digest, parse_elf
from .errors import TilefishBuildError, TilefishManifestError

if TYPE_CHECKING:
    from .aot import CompileSpec

MANIFEST_NAME = "manifest.json"


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


class Manifest:
    """The binaries of a manifest directory, verified when it is loaded."""

    def __init__(self, directory: pathlib.Path, doc: "dict[str, Any]", artifacts: "dict[tuple[str, str], Artifact]") -> None:
        self.directory = directory
        self.doc = doc
        self._artifacts = artifacts

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
        except OSError as e:
            raise TilefishManifestError(f"cannot read the manifest {str(manifest_path)!r}: {e}") from e
        try:
            doc = json.loads(text, parse_constant=_reject_constant)
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
        return cls(manifest_path.parent, doc, artifacts)

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
        return Artifact(entry, cubin)

    @property
    def artifacts(self) -> "list[Artifact]":
        return list(self._artifacts.values())

    def __contains__(self, key: "tuple[str, str]") -> bool:
        return key in self._artifacts

    def get(self, stem: str, arch: str) -> "Artifact | None":
        """The artifact of ``stem`` for ``arch`` (``"sm_103"``), or ``None``."""
        return self._artifacts.get((stem, arch))

    def built_kernel(self, spec: "CompileSpec", arch: str) -> BuiltKernel:
        """The binary for ``spec`` on ``arch``, as a :class:`~tilefish.aot.BuiltKernel` a replay can launch.

        The manifest entry must be what the spec asks for: every field the spec determines (ABI, grid, block,
        ``num_warps``, constants, buckets, preconditions, compile options and declaration digest) must equal the
        spec's. The result has no PTX (a manifest does not carry it).
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
        )

    def __repr__(self) -> str:
        return f"<Manifest {str(self.directory)!r}: {len(self._artifacts)} binaries>"
