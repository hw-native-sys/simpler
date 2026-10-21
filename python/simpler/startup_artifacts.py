# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded, content-addressed Python startup artifacts, independent of native Simpler.

The parent reads local paths once and broadcasts the resulting byte descriptions.
Every rank installs and verifies those bytes locally before explicit activation.
Installing or activating an artifact never executes its source code.
Archives must use plain USTAR regular-file/directory headers (for example,
``tarfile.USTAR_FORMAT``); PAX and GNU extensions are deliberately unsupported.
Activated prefixes support Python source and namespace packages only; sourceless
bytecode and native-extension imports beneath those prefixes are unsupported.
"""

import gzip
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import io
import json
import keyword
import os
import re
import shutil
import sys
import tarfile
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Union

MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_UNPACKED_BYTES = 256 * 1024 * 1024
MAX_MEMBERS = 4096
_PROTOCOL_VERSION = 1
_IDENTITY_FIELDS = {"protocol_version", "sha256", "module_prefix"}
_ACTIVATION_LOCK = threading.RLock()


def _validate_identity(digest, prefix):
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("artifact SHA256 digest must be 64 lowercase hex characters")
    if (
        not isinstance(prefix, str)
        or not prefix
        or any(not part.isidentifier() or keyword.iskeyword(part) for part in prefix.split("."))
    ):
        raise ValueError("artifact module_prefix must contain dotted Python identifiers")


@dataclass(frozen=True)
class StartupArtifactSpec:
    """Parent-local archive and its externally computed digest and import prefix."""

    archive_path: Union[str, Path]
    sha256: str
    module_prefix: str

    def __post_init__(self):
        _validate_identity(self.sha256, self.module_prefix)
        if not isinstance(self.archive_path, (str, Path)) or not str(self.archive_path):
            raise ValueError("artifact archive_path must be a nonempty string or Path")

    def to_manifest(self):
        return {
            "protocol_version": _PROTOCOL_VERSION,
            "archive_path": str(self.archive_path),
            "sha256": self.sha256,
            "module_prefix": self.module_prefix,
        }


@dataclass(frozen=True)
class InstalledStartupArtifact:
    """Verified content directory and its process-local import name."""

    digest: str
    root: Path
    module_prefix: str


def _description(description, field):
    if not isinstance(description, Mapping) or set(description) != _IDENTITY_FIELDS | {field}:
        raise ValueError("malformed startup artifact description")
    if type(description["protocol_version"]) is not int or description["protocol_version"] != 1:
        raise ValueError("unsupported startup artifact protocol_version")
    _validate_identity(description["sha256"], description["module_prefix"])
    return description


def _check_payload(payload, digest):
    if not isinstance(payload, bytes):
        raise ValueError("artifact payload must be bytes")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("artifact payload exceeds size limit")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("artifact SHA256 digest mismatch")


def read_startup_artifacts(descriptions):
    """Read parent-local manifests into path-free byte descriptions on rank zero."""
    result = []
    for description in descriptions:
        description = _description(description, "archive_path")
        spec = StartupArtifactSpec(description["archive_path"], description["sha256"], description["module_prefix"])
        with Path(spec.archive_path).open("rb") as source:
            payload = source.read(MAX_PAYLOAD_BYTES + 1)
        _check_payload(payload, spec.sha256)
        result.append(
            {"protocol_version": 1, "sha256": spec.sha256, "module_prefix": spec.module_prefix, "payload": payload}
        )
    return result


def _member_path(member):
    name = member.name.rstrip("/") if member.isdir() else member.name
    parts = name.split("/")
    if (
        not name
        or "\\" in name
        or ":" in name
        or any(part in ("", ".", "..") or part.endswith((".", " ")) or "\x00" in part for part in parts)
    ):
        raise ValueError("unsafe artifact archive path")
    # Keep extraction semantics identical on Windows and POSIX.
    reserved = {"con", "prn", "aux", "nul"} | {f"{stem}{i}" for stem in ("com", "lpt") for i in range(1, 10)}
    if any(part.split(".")[0].lower() in reserved for part in parts):
        raise ValueError("unsafe artifact archive path")
    return name


def _preflight_tar(raw):
    """Walk physical headers without tarfile's recursive extension processing."""
    offset, count = 0, 0
    zero = b"\0" * 512
    while offset + 512 <= len(raw):
        header = raw[offset : offset + 512]
        if header == zero:
            if raw[offset : offset + 1024] != zero * 2 or any(memoryview(raw)[offset + 1024 :]):
                raise ValueError("invalid USTAR archive terminator")
            return
        count += 1
        if count > MAX_MEMBERS:
            raise ValueError("artifact archive has too many members")
        if header[156:157] not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE):
            raise ValueError("unsupported artifact archive extension or member type; plain USTAR required")
        if header[257:265] != b"ustar\00000":
            raise ValueError("unsupported artifact archive format; plain USTAR required")
        # frombuf validates just this fixed-size header; it cannot follow extensions.
        member = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
        if member.size < 0 or (member.isdir() and member.size != 0):
            raise ValueError("invalid USTAR archive member size")
        offset += 512 + ((member.size + 511) // 512) * 512
        if offset > len(raw):
            raise ValueError("truncated USTAR archive member")
    raise ValueError("missing USTAR archive terminator")


def _archive_contents(payload, destination=None):  # noqa: PLR0912 - One bounded pass validates all archive safety gates.
    """Validate every member, optionally writing only regular files and directories."""
    # Bound gzip expansion too: tar headers, padding and extended metadata count.
    raw_limit = MAX_UNPACKED_BYTES + MAX_MEMBERS * 1024 + 10240
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as compressed:
            raw = compressed.read(raw_limit + 1)
        if len(raw) > raw_limit:
            raise ValueError("artifact expanded archive exceeds size limit")
        _preflight_tar(raw)
        entries, explicit, aliases, total = {}, set(), {}, 0
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            for count, member in enumerate(archive, 1):
                if count > MAX_MEMBERS:
                    raise ValueError("artifact archive has too many members")
                if not (member.isfile() or member.isdir()) or member.issparse():
                    raise ValueError("artifact archive allows only regular files and directories")
                name = _member_path(member)
                if name in explicit:
                    raise ValueError("duplicate artifact archive path")
                explicit.add(name)
                parts = name.split("/")
                for length in range(1, len(parts) + 1):
                    path = "/".join(parts[:length])
                    alias = path.casefold()
                    if alias in aliases and aliases[alias] != path:
                        raise ValueError("duplicate artifact archive path alias")
                    aliases[alias] = path
                    if length < len(parts):
                        if entries.get(path, {}).get("type") == "file":
                            raise ValueError("conflicting artifact archive paths")
                        entries[path] = {"type": "directory"}
                if name in entries and (member.isfile() or entries[name]["type"] != "directory"):
                    raise ValueError("conflicting artifact archive paths")
                target = destination / name if destination is not None else None
                if member.isdir():
                    entries[name] = {"type": "directory"}
                    if target is not None:
                        target.mkdir(parents=True, exist_ok=True)
                    continue
                total += member.size
                if member.size < 0 or total > MAX_UNPACKED_BYTES:
                    raise ValueError("artifact unpacked content exceeds size limit")
                digest = hashlib.sha256()
                if target is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source:
                    output = target.open("xb") if target is not None else None
                    try:
                        remaining = member.size
                        while remaining:
                            chunk = source.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError("truncated artifact archive file")
                            digest.update(chunk)
                            if output is not None:
                                output.write(chunk)
                            remaining -= len(chunk)
                    finally:
                        if output is not None:
                            output.close()
                entries[name] = {"type": "file", "size": member.size, "sha256": digest.hexdigest()}
        return entries
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ValueError("invalid artifact tar.gz archive") from exc


def _validate_cache(directory, identity):  # noqa: PLR0912 - Identity and complete-tree checks share one corruption boundary.
    """Never trust cached metadata alone; hash and compare the actual complete tree."""
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("corrupt artifact cache directory")
        if {item.name for item in directory.iterdir()} != {"content", "identity.json"}:
            raise ValueError("corrupt artifact cache layout")
        metadata = directory / "identity.json"
        if metadata.is_symlink() or not metadata.is_file():
            raise ValueError("corrupt artifact cache identity")
        expected = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        with metadata.open("rb") as source:
            if source.read(len(expected) + 1) != expected:
                raise ValueError("corrupt artifact cache identity")
        content, found = directory / "content", {}
        if content.is_symlink() or not content.is_dir():
            raise ValueError("corrupt artifact cache content directory")
        pending = [content]
        while pending:
            for item in pending.pop().iterdir():
                name = item.relative_to(content).as_posix()
                entry = identity["entries"].get(name)
                if item.is_symlink() or entry is None:
                    raise ValueError("corrupt artifact cache content")
                if entry["type"] == "directory" and item.is_dir():
                    pending.append(item)
                elif entry["type"] == "file" and item.is_file():
                    if item.stat().st_size != entry["size"]:
                        raise ValueError("corrupt artifact cache file size")
                    digest = hashlib.sha256()
                    with item.open("rb") as source:
                        remaining = entry["size"]
                        while remaining:
                            chunk = source.read(min(remaining, 1024 * 1024))
                            if not chunk:
                                raise ValueError("corrupt artifact cache file")
                            digest.update(chunk)
                            remaining -= len(chunk)
                        if source.read(1) or digest.hexdigest() != entry["sha256"]:
                            raise ValueError("corrupt artifact cache file digest")
                else:
                    raise ValueError("corrupt artifact cache entry type")
                found[name] = entry
        if found != identity["entries"]:
            raise ValueError("corrupt artifact cache missing content")
    except OSError as exc:
        raise ValueError("cannot validate artifact cache") from exc


def install_startup_artifacts(payload_descriptions, cache_root):
    """Verify and atomically install payloads on one rank; concurrent installs are safe."""
    cache_root = Path(cache_root).absolute()
    result = []
    for description in payload_descriptions:
        description = _description(description, "payload")
        payload, digest = description["payload"], description["sha256"]
        _check_payload(payload, digest)
        entries = _archive_contents(payload)
        identity = {"protocol_version": 1, "sha256": digest, "entries": entries}
        cache_root.mkdir(parents=True, exist_ok=True)
        if cache_root.is_symlink():
            raise ValueError("artifact cache root must not be a symlink")
        directory = cache_root / digest
        if not os.path.lexists(directory):
            staging = Path(tempfile.mkdtemp(prefix=f".{digest}-", dir=cache_root))
            try:
                content = staging / "content"
                content.mkdir()
                _archive_contents(payload, content)
                (staging / "identity.json").write_text(
                    json.dumps(identity, sort_keys=True, separators=(",", ":")), encoding="utf-8"
                )
                try:
                    staging.rename(directory)
                except OSError:
                    if not os.path.lexists(directory):
                        raise
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        _validate_cache(directory, identity)
        result.append(InstalledStartupArtifact(digest, directory / "content", description["module_prefix"]))
    return result


class _SourceLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # Cached bytecode is outside the verified archive and must never be trusted.
        return self.source_to_code(self.get_data(self.path), self.path)


class _ArtifactFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.bindings = {}

    def find_spec(self, fullname, path=None, target=None):
        for prefix, (_, root) in self.bindings.items():
            if not fullname.startswith(prefix + "."):
                continue
            relative = fullname[len(prefix) + 1 :].split(".")
            location = root.joinpath(*relative)
            for source, package in ((location / "__init__.py", True), (location.with_suffix(".py"), False)):
                if source.is_file():
                    return importlib.util.spec_from_file_location(
                        fullname,
                        source,
                        loader=_SourceLoader(fullname, str(source)),
                        submodule_search_locations=[str(location)] if package else None,
                    )
            if location.is_dir():
                spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
                spec.submodule_search_locations = [str(location)]
                return spec
            # A matched prefix is authoritative. PathFinder must not fall back to
            # sourceless bytecode or native extensions through namespace.__path__.
            raise ModuleNotFoundError(f"no supported source module in artifact: {fullname}", name=fullname)
        return None


_FINDER = _ArtifactFinder()


def activate_startup_artifacts(installed):
    """Bind generic namespace prefixes for later ordinary Python imports."""
    with _ACTIVATION_LOCK:
        for artifact in installed:
            _validate_identity(artifact.digest, artifact.module_prefix)
            prefix = artifact.module_prefix
            binding = _FINDER.bindings.get(prefix)
            if binding is not None:
                if binding[0] != artifact.digest:
                    raise ValueError("artifact module prefix is already bound to another artifact")
                continue
            parts = prefix.split(".")
            for length in range(1, len(parts) + 1):
                name = ".".join(parts[:length])
                existing = sys.modules.get(name)
                if existing is not None and (
                    not getattr(existing, "__startup_artifact_namespace__", False)
                    or name in _FINDER.bindings
                    or length == len(parts)
                ):
                    raise ValueError("artifact module prefix collides with an already bound module")
            if any(name.startswith(prefix + ".") for name in sys.modules):
                raise ValueError("artifact module prefix collides with an already bound descendant")
            root = Path(artifact.root)
            for length in range(1, len(parts) + 1):
                name = ".".join(parts[:length])
                if name not in sys.modules:
                    module = ModuleType(name)
                    module.__package__ = name
                    module.__path__ = [str(root)] if name == prefix else []
                    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
                    module.__spec__.submodule_search_locations = module.__path__
                    module.__startup_artifact_namespace__ = True
                    sys.modules[name] = module
                    if length > 1:
                        setattr(sys.modules[".".join(parts[: length - 1])], parts[length - 1], module)
            _FINDER.bindings[prefix] = (artifact.digest, root)
        if _FINDER.bindings and _FINDER not in sys.meta_path:
            sys.meta_path.insert(0, _FINDER)
