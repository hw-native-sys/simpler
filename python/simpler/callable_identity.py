# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Callable identity helpers for Worker callable registration."""

from __future__ import annotations

import ctypes
import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any, Literal

from .task_interface import ArgDirection, ChipCallable

CALLABLE_DESCRIPTOR_SCHEMA_VERSION = 1
CALLABLE_HASH_DIGEST_BYTES = 32
CALLABLE_KIND_CHIP = 1
CALLABLE_KIND_PYTHON_SERIALIZED = 2
CALLABLE_KIND_PYTHON_IMPORT = 3
TARGET_NAMESPACE_LOCAL_CHIP = "LOCAL_CHIP"
TARGET_NAMESPACE_LOCAL_PYTHON = "LOCAL_PYTHON"
TARGET_NAMESPACE_REMOTE_TASK_DISPATCHER = "REMOTE_TASK_DISPATCHER"

CallableKindName = Literal["CHIP_CALLABLE", "PYTHON_SERIALIZED", "PYTHON_IMPORT"]
TargetNamespaceName = Literal["LOCAL_CHIP", "LOCAL_PYTHON", "REMOTE_TASK_DISPATCHER"]

__all__ = [
    "CALLABLE_HASH_DIGEST_BYTES",
    "CallableHandle",
    "CallableKindName",
    "TargetNamespaceName",
    "build_chip_callable_descriptor",
    "build_chip_signature_schema",
    "build_python_import_descriptor",
    "build_python_serialized_descriptor",
    "compute_callable_hashid",
    "hashid_to_digest",
    "parse_python_import_target",
    "parse_python_callable_payload",
    "validate_hashid",
]

_PY_CALLABLE_MAGIC = b"SPYC"
_PY_CALLABLE_VERSION = 1
_PY_CALLABLE_SERIALIZER_CLOUDPICKLE = 1
_PY_CALLABLE_HEADER = struct.Struct("<4sBBHQ")


def _pack_u32(value: int) -> bytes:
    return struct.pack("<I", int(value))


def _pack_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return _pack_u32(len(data)) + data


def _pack_bytes(value: bytes) -> bytes:
    return _pack_u32(len(value)) + value


def _validate_dotted_ident(value: str, *, field_name: str) -> None:
    parts = value.split(".")
    if any(not part for part in parts):
        raise ValueError(f"{field_name} must contain non-empty dot-separated identifiers")
    if any(not part.isidentifier() for part in parts):
        raise ValueError(f"{field_name} contains a non-identifier component: {value!r}")


def _sha256_digest(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _sha256_hashid(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _platform_arch(platform: str) -> str:
    if platform in ("a2a3", "a2a3sim"):
        return "a2a3"
    if platform in ("a5", "a5sim"):
        return "a5"
    if not platform:
        return ""
    raise ValueError(f"Unknown platform for callable descriptor: {platform}")


def _chip_callable_bytes(target: ChipCallable) -> bytes:
    return ctypes.string_at(int(target.buffer_ptr()), int(target.buffer_size()))


def _arg_direction_value(direction: ArgDirection) -> int:
    return int(direction.value if hasattr(direction, "value") else direction)


def build_chip_signature_schema(target: ChipCallable) -> bytes:
    data = bytearray()
    data += _pack_u32(1)
    sig_count = int(target.sig_count)
    data += _pack_u32(sig_count)
    for i in range(sig_count):
        data += _pack_u32(_arg_direction_value(target.sig(i)))
    return bytes(data)


def build_chip_callable_descriptor(*, target: ChipCallable, platform: str = "", runtime: str = "") -> bytes:
    blob = _chip_callable_bytes(target)
    signature_digest = _sha256_digest(build_chip_signature_schema(target))
    data = bytearray()
    data += _pack_u32(CALLABLE_DESCRIPTOR_SCHEMA_VERSION)
    data += _pack_u32(CALLABLE_KIND_CHIP)
    data += _pack_string(_platform_arch(platform))
    data += _pack_string(platform)
    data += _pack_string(runtime)
    data += _pack_bytes(_sha256_digest(blob))
    data += _pack_bytes(signature_digest)
    return bytes(data)


def parse_python_callable_payload(payload: bytes) -> tuple[int, int, bytes]:
    if len(payload) < _PY_CALLABLE_HEADER.size:
        raise ValueError(f"python callable payload too small: {len(payload)} bytes")
    magic, version, serializer, flags, payload_size = _PY_CALLABLE_HEADER.unpack_from(payload, 0)
    if magic != _PY_CALLABLE_MAGIC:
        raise ValueError(f"invalid python callable payload magic: {magic!r}")
    if version != _PY_CALLABLE_VERSION:
        raise ValueError(f"unsupported python callable payload version: {version}")
    if serializer != _PY_CALLABLE_SERIALIZER_CLOUDPICKLE:
        raise ValueError(f"unsupported python callable serializer: {serializer}")
    if flags != 0:
        raise ValueError(f"unsupported python callable payload flags: {flags}")
    expected_size = _PY_CALLABLE_HEADER.size + int(payload_size)
    if expected_size > len(payload):
        raise ValueError(f"python callable payload size mismatch: header={payload_size}, payload={len(payload)}")
    return int(version), int(serializer), payload[_PY_CALLABLE_HEADER.size : expected_size]


def build_python_serialized_descriptor(payload: bytes) -> bytes:
    version, serializer, serializer_payload = parse_python_callable_payload(payload)
    data = bytearray()
    data += _pack_u32(CALLABLE_DESCRIPTOR_SCHEMA_VERSION)
    data += _pack_u32(CALLABLE_KIND_PYTHON_SERIALIZED)
    data += _pack_u32(version)
    data += _pack_u32(serializer)
    data += _pack_bytes(_sha256_digest(serializer_payload))
    return bytes(data)


def parse_python_import_target(target: str) -> tuple[str, str]:
    if not isinstance(target, str):
        raise TypeError("RemoteCallable target must be a string")
    normalized = target.strip(" \t\r\n")
    if normalized.count(":") != 1:
        raise ValueError("RemoteCallable target must have exactly one ':' separator")
    module, qualname = normalized.split(":", 1)
    if not module or not qualname:
        raise ValueError("RemoteCallable module and qualname must be non-empty")
    if module.startswith("."):
        raise ValueError("RemoteCallable module must be absolute")
    if "<locals>" in qualname.split("."):
        raise ValueError("RemoteCallable qualname must not contain <locals>")
    _validate_dotted_ident(module, field_name="RemoteCallable module")
    _validate_dotted_ident(qualname, field_name="RemoteCallable qualname")
    return module, qualname


def build_python_import_descriptor(module: str, qualname: str) -> bytes:
    parse_python_import_target(f"{module}:{qualname}")
    data = bytearray()
    data += _pack_u32(CALLABLE_DESCRIPTOR_SCHEMA_VERSION)
    data += _pack_u32(CALLABLE_KIND_PYTHON_IMPORT)
    data += _pack_string(module)
    data += _pack_string(qualname)
    return bytes(data)


def compute_callable_hashid(descriptor: bytes) -> str:
    return _sha256_hashid(descriptor)


def hashid_to_digest(hashid: str) -> bytes:
    validate_hashid(hashid)
    return bytes.fromhex(hashid.removeprefix("sha256:"))


def validate_hashid(hashid: str) -> None:
    if not isinstance(hashid, str):
        raise TypeError("CallableHandle hashid must be a string")
    if not hashid.startswith("sha256:"):
        raise ValueError("HASHID_FORMAT_INVALID: hashid must start with 'sha256:'")
    hex_part = hashid[len("sha256:") :]
    if len(hex_part) != 64:
        raise ValueError("HASHID_FORMAT_INVALID: sha256 digest must be 64 lowercase hex characters")
    if hex_part.lower() != hex_part:
        raise ValueError("HASHID_FORMAT_INVALID: sha256 digest must use lowercase hex")
    try:
        bytes.fromhex(hex_part)
    except ValueError as exc:
        raise ValueError("HASHID_FORMAT_INVALID: sha256 digest contains non-hex characters") from exc


class CallableHandle:
    """Opaque public token returned by ``Worker.register``."""

    __slots__ = ("hashid", "kind", "target_namespace", "_digest", "_handle_id", "_owner_id")

    def __init__(
        self,
        hashid: str,
        kind: CallableKindName,
        target_namespace: TargetNamespaceName,
    ) -> None:
        validate_hashid(hashid)
        self.hashid = hashid
        self.kind = kind
        self.target_namespace = target_namespace
        self._digest = hashid_to_digest(hashid)
        self._handle_id = -1
        self._owner_id: str | None = None
        self._validate_public_fields()

    @classmethod
    def _from_registration(
        cls,
        *,
        hashid: str,
        kind: CallableKindName,
        target_namespace: TargetNamespaceName,
        handle_id: int,
        owner_id: str,
    ) -> CallableHandle:
        handle = cls(hashid, kind, target_namespace)
        handle._handle_id = int(handle_id)
        handle._owner_id = owner_id
        return handle

    def _validate_public_fields(self) -> None:
        if self.kind not in ("CHIP_CALLABLE", "PYTHON_SERIALIZED", "PYTHON_IMPORT"):
            raise ValueError(f"CALLABLE_KIND_UNSUPPORTED: {self.kind}")
        if self.target_namespace not in (
            TARGET_NAMESPACE_LOCAL_CHIP,
            TARGET_NAMESPACE_LOCAL_PYTHON,
            TARGET_NAMESPACE_REMOTE_TASK_DISPATCHER,
        ):
            raise ValueError(f"unsupported callable target namespace: {self.target_namespace}")

    @property
    def digest(self) -> bytes:
        return self._digest


@dataclass
class _CallableIdentityState:
    hashid: str
    digest: bytes
    kind: CallableKindName
    target_namespace: TargetNamespaceName
    descriptor: bytes
    payload_digest: bytes
    slot_id: int
    target: Any
    ref_count: int = 0
    state: str = "INSTALLED"
    eligible_worker_ids: tuple[int, ...] = field(default_factory=tuple)
