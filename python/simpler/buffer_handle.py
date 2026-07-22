# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Owner/consumer side of the P1-B BufferHandle ABI (Python mirror of ``buffer_handle.h``).

A ``BufferHandle`` is the owner's registry object for one shared backing (identity + backend + the
POSIX shm that backs it). ``create_host_shared_buffer`` mints one; ``BufferHandle.to_descriptor``
serializes the wire ``BufferHandleDescriptor`` sent once per edge in the export handshake. An
``ImportRegistry`` on the consumer side maps a descriptor into this process and resolves a
``CanonicalIdentity`` to a local base address — the typed successor of ``host_buf_table`` /
``_rewrite_blob_host_addrs``.

Struct formats mirror ``buffer_handle.h`` byte for byte; their sizes are asserted at import against
the constants the ``_task_interface`` binding exports, so a layout drift fails loudly here.
"""

from __future__ import annotations

import ctypes
import enum
import os
import struct
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

from _task_interface import (  # pyright: ignore[reportMissingImports]
    BACKEND_TOKEN_BYTES,
    BUFFER_ABI_VERSION,
    BUFFER_HANDLE_DESCRIPTOR_BYTES,
    CANONICAL_IDENTITY_BYTES,
    MAX_WORKER_PATH_DEPTH,
    OWNER_WORKER_PATH_BYTES,
)


class AddressSpace(enum.IntEnum):
    HOST = 0
    DEVICE = 1


class Visibility(enum.IntEnum):
    PRIVATE = 0
    SHARED = 1


class AccessMode(enum.IntEnum):
    READ = 0
    WRITE = 1
    READWRITE = 2


class BackendKind(enum.IntEnum):
    INVALID = 0
    FORK_SHM = 1
    POSIX_SHM = 2
    VMM_WINDOW = 3
    DEVICE_MALLOC = 4
    REMOTE_SIDECAR = 5


# ---------------------------------------------------------------------------
# Wire struct formats — mirror buffer_handle.h; sizes pinned to the binding.
# ---------------------------------------------------------------------------
#
# CanonicalIdentity (40 B): owner_instance_id u64, buffer_id u64,
#   owner_worker_path{ depth u8, _reserved[3], hop[MAX_WORKER_PATH_DEPTH] u32 }, generation u32.
_CANONICAL_IDENTITY = struct.Struct(f"<QQB3x{MAX_WORKER_PATH_DEPTH}II")
# BufferHandleDescriptor (128 B) = prefix(8) + CanonicalIdentity(40) + suffix(80).
_DESC_PREFIX = struct.Struct("<IBBBB")  # abi_version, address_space, visibility, access, backend_kind
_DESC_SUFFIX = struct.Struct(f"<QQ{BACKEND_TOKEN_BYTES}s")  # nbytes, backend_handle, token

assert _CANONICAL_IDENTITY.size == CANONICAL_IDENTITY_BYTES, (_CANONICAL_IDENTITY.size, CANONICAL_IDENTITY_BYTES)
assert _CANONICAL_IDENTITY.size == OWNER_WORKER_PATH_BYTES + 20, "identity = 2*u64 + path + u32"
assert _DESC_PREFIX.size + _CANONICAL_IDENTITY.size + _DESC_SUFFIX.size == BUFFER_HANDLE_DESCRIPTOR_BYTES, (
    "BufferHandleDescriptor layout drift vs buffer_handle.h"
)


@dataclass(frozen=True)
class CanonicalIdentity:
    """Globally-unique allocation identity; the key of every import registry."""

    owner_instance_id: int
    buffer_id: int
    owner_worker_path: tuple[int, ...]  # child index at each hop from root; len == tree depth
    generation: int = 0

    def __post_init__(self) -> None:
        if len(self.owner_worker_path) > MAX_WORKER_PATH_DEPTH:
            raise ValueError(
                f"owner_worker_path depth {len(self.owner_worker_path)} exceeds MAX_WORKER_PATH_DEPTH "
                f"{MAX_WORKER_PATH_DEPTH}"
            )

    def pack(self) -> bytes:
        depth = len(self.owner_worker_path)
        hops = list(self.owner_worker_path) + [0] * (MAX_WORKER_PATH_DEPTH - depth)
        return _CANONICAL_IDENTITY.pack(self.owner_instance_id, self.buffer_id, depth, *hops, self.generation)

    @classmethod
    def unpack(cls, raw: bytes) -> CanonicalIdentity:
        fields = _CANONICAL_IDENTITY.unpack(raw)
        owner_instance_id, buffer_id, depth = fields[0], fields[1], fields[2]
        hops = fields[3 : 3 + MAX_WORKER_PATH_DEPTH]
        generation = fields[3 + MAX_WORKER_PATH_DEPTH]
        if depth > MAX_WORKER_PATH_DEPTH:
            raise ValueError(f"owner_worker_path depth {depth} exceeds MAX_WORKER_PATH_DEPTH {MAX_WORKER_PATH_DEPTH}")
        return cls(owner_instance_id, buffer_id, tuple(hops[:depth]), generation)


@dataclass(frozen=True)
class BufferHandleDescriptor:
    """The export-handshake wire payload — the owner's handle projected to a flat, versioned blob."""

    identity: CanonicalIdentity
    address_space: AddressSpace
    visibility: Visibility
    access: AccessMode
    backend_kind: BackendKind
    nbytes: int
    token: str = ""  # NUL-terminated shm name for FORK_SHM/POSIX_SHM; "" otherwise
    backend_handle: int = 0  # VMM shareable-handle / device ptr for VMM_WINDOW/DEVICE_MALLOC; 0 otherwise

    def pack(self) -> bytes:
        token = self.token.encode("utf-8")
        if len(token) >= BACKEND_TOKEN_BYTES:
            raise ValueError(f"backend token {self.token!r} too long ({len(token)} >= {BACKEND_TOKEN_BYTES})")
        prefix = _DESC_PREFIX.pack(
            BUFFER_ABI_VERSION,
            int(self.address_space),
            int(self.visibility),
            int(self.access),
            int(self.backend_kind),
        )
        suffix = _DESC_SUFFIX.pack(self.nbytes, self.backend_handle, token)
        return prefix + self.identity.pack() + suffix

    @classmethod
    def unpack(cls, raw: bytes) -> BufferHandleDescriptor:
        if len(raw) < BUFFER_HANDLE_DESCRIPTOR_BYTES:
            raise ValueError(f"descriptor too small: {len(raw)} < {BUFFER_HANDLE_DESCRIPTOR_BYTES}")
        abi_version, address_space, visibility, access, backend_kind = _DESC_PREFIX.unpack_from(raw, 0)
        if abi_version != BUFFER_ABI_VERSION:
            raise ValueError(f"unknown BufferHandle abi_version {abi_version} (expected {BUFFER_ABI_VERSION})")
        identity = CanonicalIdentity.unpack(raw[_DESC_PREFIX.size : _DESC_PREFIX.size + _CANONICAL_IDENTITY.size])
        nbytes, backend_handle, token = _DESC_SUFFIX.unpack_from(raw, _DESC_PREFIX.size + _CANONICAL_IDENTITY.size)
        token_str = token.split(b"\x00", 1)[0].decode("utf-8")
        return cls(
            identity=identity,
            address_space=AddressSpace(address_space),
            visibility=Visibility(visibility),
            access=AccessMode(access),
            backend_kind=BackendKind(backend_kind),
            nbytes=nbytes,
            token=token_str,
            backend_handle=backend_handle,
        )


def mint_owner_instance_id() -> int:
    """A fresh 64-bit nonce, unique per owner incarnation (defends canonical identity against ABA)."""
    return int.from_bytes(os.urandom(8), "little")


def _shm_base_addr(shm: SharedMemory) -> int:
    """Mapped base address of ``shm``; valid until ``shm.close()``."""
    view = shm.buf
    assert view is not None
    exporter = ctypes.c_char.from_buffer(view)
    addr = ctypes.addressof(exporter)
    del exporter
    return addr


@dataclass
class BufferHandle:
    """Owner-side registry object for one shared backing; owns the POSIX shm that backs it."""

    identity: CanonicalIdentity
    address_space: AddressSpace
    visibility: Visibility
    access: AccessMode
    backend_kind: BackendKind
    nbytes: int
    token: str = ""
    backend_handle: int = 0
    shm: SharedMemory | None = None
    base: int = 0

    def to_descriptor(self) -> BufferHandleDescriptor:
        return BufferHandleDescriptor(
            identity=self.identity,
            address_space=self.address_space,
            visibility=self.visibility,
            access=self.access,
            backend_kind=self.backend_kind,
            nbytes=self.nbytes,
            token=self.token,
            backend_handle=self.backend_handle,
        )

    def close(self) -> None:
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None


def create_host_shared_buffer(
    nbytes: int,
    owner_instance_id: int,
    buffer_id: int,
    owner_worker_path: tuple[int, ...] = (),
    generation: int = 0,
    visibility: Visibility = Visibility.SHARED,
    access: AccessMode = AccessMode.READWRITE,
) -> BufferHandle:
    """Allocate a POSIX-shm host backing and wrap it as an owner ``BufferHandle`` (backend POSIX_SHM)."""
    if nbytes <= 0:
        raise ValueError(f"create_host_shared_buffer: nbytes must be positive, got {nbytes}")
    shm = SharedMemory(create=True, size=nbytes)
    identity = CanonicalIdentity(owner_instance_id, buffer_id, tuple(owner_worker_path), generation)
    return BufferHandle(
        identity=identity,
        address_space=AddressSpace.HOST,
        visibility=visibility,
        access=access,
        backend_kind=BackendKind.POSIX_SHM,
        nbytes=nbytes,
        token=shm.name,
        backend_handle=0,
        shm=shm,
        base=_shm_base_addr(shm),
    )


@dataclass
class ImportedBuffer:
    """A handle materialized into the consumer's address space: identity -> local base."""

    identity: CanonicalIdentity
    base: int
    nbytes: int
    shm: SharedMemory | None = None  # the consumer's own mapping for shm backends


class ImportRegistry:
    """Per-consumer-endpoint registry: register an export descriptor, resolve identity -> local base.

    Keyed by the packed canonical identity so lookups are exact (never a numeric-range guess). Host
    shm backends are mapped into this process on register; a later ``BufferRef`` referencing the
    handle resolves to the local base without re-sending the descriptor.
    """

    def __init__(self) -> None:
        self._by_identity: dict[bytes, ImportedBuffer] = {}

    def register(self, descriptor: BufferHandleDescriptor | bytes) -> ImportedBuffer:
        desc = BufferHandleDescriptor.unpack(descriptor) if isinstance(descriptor, (bytes, bytearray)) else descriptor
        if desc.backend_kind in (BackendKind.POSIX_SHM, BackendKind.FORK_SHM):
            shm = SharedMemory(name=desc.token)
            imported = ImportedBuffer(desc.identity, _shm_base_addr(shm), desc.nbytes, shm)
        elif desc.backend_kind == BackendKind.REMOTE_SIDECAR:
            raise ValueError("ImportRegistry: REMOTE_SIDECAR backend is reserved for P2")
        else:
            raise NotImplementedError(f"ImportRegistry: backend {desc.backend_kind!r} not supported in P1-B")
        key = desc.identity.pack()
        prior = self._by_identity.pop(key, None)
        if prior is not None and prior.shm is not None:
            prior.shm.close()
        self._by_identity[key] = imported
        return imported

    def resolve(self, identity: CanonicalIdentity) -> ImportedBuffer:
        imported = self._by_identity.get(identity.pack())
        if imported is None:
            raise KeyError(f"ImportRegistry: no handle registered for {identity}")
        return imported

    def unregister(self, identity: CanonicalIdentity) -> None:
        imported = self._by_identity.pop(identity.pack(), None)
        if imported is not None and imported.shm is not None:
            imported.shm.close()

    def close(self) -> None:
        for imported in self._by_identity.values():
            if imported.shm is not None:
                imported.shm.close()
        self._by_identity.clear()
