# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Owner/consumer side of the BufferHandle ABI (Python mirror of ``buffer_handle.h``).

Implements the frozen logical schema in ``.docs/L3-new/worker-memory-model/bufferhandle-abi.md``:
16-byte opaque ``owner_instance_id``, length-delimited UTF-8 ``owner_worker_path`` ("L4/L3[2]/L2[5]"),
and a versioned length-delimited ``backend`` body. Struct formats mirror ``buffer_handle.h`` byte for
byte; their sizes are asserted at import against the constants the ``_task_interface`` binding
exports, so a layout drift fails loudly here.
"""

from __future__ import annotations

import ctypes
import enum
import os
import struct
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

from _task_interface import (  # pyright: ignore[reportMissingImports]
    BUFFER_ABI_VERSION,
    BUFFER_DESCRIPTOR_VERSION,
    BUFFER_HANDLE_DESCRIPTOR_BYTES,
    BUFFER_REF_BYTES,
    BUFFERREF_BLOB_HEADER_BYTES,
    CANONICAL_IDENTITY_BYTES,
    DESC_MAX_BYTES,
    MAX_TENSOR_DIMS,
    OWNER_INSTANCE_ID_BYTES,
    PATH_MAX_BYTES,
    bufferref_blob_descriptors,
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
    FORK_SHM = 0
    POSIX_SHM = 1
    VMM_WINDOW = 2
    REMOTE_SIDECAR = 3
    DEVICE_MALLOC = 4


# ---------------------------------------------------------------------------
# Wire struct formats — mirror buffer_handle.h; sizes pinned to the binding.
# ---------------------------------------------------------------------------
#
# CanonicalIdentity (96 B): owner_instance_id[16] opaque, buffer_id u64, generation u32,
#   path_len u16, _pad[2], owner_worker_path[PATH_MAX_BYTES] UTF-8.
_CANONICAL_IDENTITY = struct.Struct(f"<{OWNER_INSTANCE_ID_BYTES}sQIH2x{PATH_MAX_BYTES}s")
# BufferHandleDescriptor (216 B) = prefix(8) + CanonicalIdentity(96) + suffix.
_DESC_PREFIX = struct.Struct("<HBBBBBx")  # abi_version, address_space, visibility, access, backend_kind, desc_version
_DESC_SUFFIX = struct.Struct(f"<QH6x{DESC_MAX_BYTES}s")  # nbytes, body_len, body

assert _CANONICAL_IDENTITY.size == CANONICAL_IDENTITY_BYTES, (_CANONICAL_IDENTITY.size, CANONICAL_IDENTITY_BYTES)
assert _DESC_PREFIX.size + _CANONICAL_IDENTITY.size + _DESC_SUFFIX.size == BUFFER_HANDLE_DESCRIPTOR_BYTES, (
    "BufferHandleDescriptor layout drift vs buffer_handle.h"
)


@dataclass(frozen=True)
class CanonicalIdentity:
    """Globally-unique allocation identity; the key of every import registry.

    ``owner_instance_id`` is 16 opaque bytes (bytewise-compared). ``owner_worker_path`` is a bounded
    UTF-8 tree path such as ``"L4/L3[2]/L2[5]"``.
    """

    owner_instance_id: bytes
    buffer_id: int
    owner_worker_path: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        if len(self.owner_instance_id) != OWNER_INSTANCE_ID_BYTES:
            raise ValueError(
                f"owner_instance_id must be {OWNER_INSTANCE_ID_BYTES} bytes, got {len(self.owner_instance_id)}"
            )
        if len(self.owner_worker_path.encode("utf-8")) > PATH_MAX_BYTES:
            raise ValueError(f"owner_worker_path exceeds PATH_MAX_BYTES ({PATH_MAX_BYTES})")

    def pack(self) -> bytes:
        path = self.owner_worker_path.encode("utf-8")
        return _CANONICAL_IDENTITY.pack(self.owner_instance_id, self.buffer_id, self.generation, len(path), path)

    @classmethod
    def unpack(cls, raw: bytes) -> CanonicalIdentity:
        owner_instance_id, buffer_id, generation, path_len, path = _CANONICAL_IDENTITY.unpack(raw)
        if path_len > PATH_MAX_BYTES:
            raise ValueError(f"owner_worker_path length {path_len} exceeds PATH_MAX_BYTES ({PATH_MAX_BYTES})")
        return cls(owner_instance_id, buffer_id, path[:path_len].decode("utf-8"), generation)


@dataclass(frozen=True)
class BufferHandleDescriptor:
    """The self-describing handle payload — embedded whole in every BufferRef built over the handle.

    ``body`` is the per-backend materialization (POSIX/fork shm name UTF-8, VMM handle bytes, ...).
    """

    identity: CanonicalIdentity
    address_space: AddressSpace
    visibility: Visibility
    access: AccessMode
    backend_kind: BackendKind
    nbytes: int
    body: bytes = b""

    def pack(self) -> bytes:
        if len(self.body) > DESC_MAX_BYTES:
            raise ValueError(f"backend body exceeds DESC_MAX_BYTES ({DESC_MAX_BYTES})")
        prefix = _DESC_PREFIX.pack(
            BUFFER_ABI_VERSION,
            int(self.address_space),
            int(self.visibility),
            int(self.access),
            int(self.backend_kind),
            BUFFER_DESCRIPTOR_VERSION,
        )
        suffix = _DESC_SUFFIX.pack(self.nbytes, len(self.body), self.body)
        return prefix + self.identity.pack() + suffix

    @classmethod
    def unpack(cls, raw: bytes) -> BufferHandleDescriptor:
        if len(raw) < BUFFER_HANDLE_DESCRIPTOR_BYTES:
            raise ValueError(f"descriptor too small: {len(raw)} < {BUFFER_HANDLE_DESCRIPTOR_BYTES}")
        abi_version, address_space, visibility, access, backend_kind, descriptor_version = _DESC_PREFIX.unpack_from(
            raw, 0
        )
        if abi_version != BUFFER_ABI_VERSION:
            raise ValueError(f"unknown BufferHandle abi_version {abi_version} (expected {BUFFER_ABI_VERSION})")
        if descriptor_version != BUFFER_DESCRIPTOR_VERSION:
            raise ValueError(f"unknown backend descriptor_version {descriptor_version}")
        identity = CanonicalIdentity.unpack(raw[_DESC_PREFIX.size : _DESC_PREFIX.size + _CANONICAL_IDENTITY.size])
        nbytes, body_len, body = _DESC_SUFFIX.unpack_from(raw, _DESC_PREFIX.size + _CANONICAL_IDENTITY.size)
        return cls(
            identity=identity,
            address_space=AddressSpace(address_space),
            visibility=Visibility(visibility),
            access=AccessMode(access),
            backend_kind=BackendKind(backend_kind),
            nbytes=nbytes,
            body=bytes(body[:body_len]),
        )


# BufferRef (272 B): BufferHandleDescriptor(216) + byte_offset u64, ndims u32, shapes[MAX] u32,
#   strides[MAX] u32, dtype u8, _pad[3].
_BUFFER_REF_TAIL = struct.Struct(f"<QI{MAX_TENSOR_DIMS}I{MAX_TENSOR_DIMS}IB3x")
assert BUFFER_HANDLE_DESCRIPTOR_BYTES + _BUFFER_REF_TAIL.size == BUFFER_REF_BYTES, "BufferRef layout drift"

# BufferRef blob envelope (16 B): abi_version u32, ref_count i32, scalar_count i32, reserved u32.
_BUFFERREF_BLOB_HEADER = struct.Struct("<IiiI")
assert _BUFFERREF_BLOB_HEADER.size == BUFFERREF_BLOB_HEADER_BYTES, "BufferRef blob header drift"


@dataclass(frozen=True)
class BufferRef:
    """The blob-carried wire element: a full embedded handle descriptor + a strided view onto it.

    Self-describing — the consumer materializes ``handle`` on first receipt (no prior handshake),
    keyed by ``handle.identity``. Carries no materialized address.
    """

    handle: BufferHandleDescriptor
    byte_offset: int
    shapes: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: int  # DataType enum value

    def __post_init__(self) -> None:
        if not 0 < len(self.shapes) <= MAX_TENSOR_DIMS:
            raise ValueError(f"BufferRef ndims must be in [1, {MAX_TENSOR_DIMS}], got {len(self.shapes)}")
        if len(self.strides) != len(self.shapes):
            raise ValueError("BufferRef shapes and strides must have equal length")

    def pack(self) -> bytes:
        ndims = len(self.shapes)
        shapes = list(self.shapes) + [0] * (MAX_TENSOR_DIMS - ndims)
        strides = list(self.strides) + [0] * (MAX_TENSOR_DIMS - ndims)
        tail = _BUFFER_REF_TAIL.pack(self.byte_offset, ndims, *shapes, *strides, self.dtype)
        return self.handle.pack() + tail


def pack_bufferref_blob(refs: list[BufferRef], scalars: tuple[int, ...] = ()) -> bytes:
    """Serialize refs + scalars into the versioned BufferRef blob (mirror of write_bufferref_blob)."""
    header = _BUFFERREF_BLOB_HEADER.pack(BUFFER_ABI_VERSION, len(refs), len(scalars), 0)
    body = b"".join(ref.pack() for ref in refs)
    tail = struct.pack(f"<{len(scalars)}Q", *scalars) if scalars else b""
    return header + body + tail


def mint_owner_instance_id() -> bytes:
    """A fresh 16-byte opaque nonce, unique per owner incarnation (defends identity against ABA)."""
    return os.urandom(OWNER_INSTANCE_ID_BYTES)


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
    body: bytes = b""
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
            body=self.body,
        )

    def ref(
        self,
        shapes: tuple[int, ...],
        strides: tuple[int, ...],
        dtype: int,
        byte_offset: int = 0,
    ) -> BufferRef:
        """A self-describing BufferRef viewing this handle: embeds the full descriptor + the view.

        The consumer materializes it with no prior handshake. ``strides`` are in elements, row-major,
        strictly positive; ``byte_offset`` must be a multiple of the dtype size (checked at
        materialization).
        """
        return BufferRef(
            handle=self.to_descriptor(),
            byte_offset=byte_offset,
            shapes=tuple(shapes),
            strides=tuple(strides),
            dtype=int(dtype),
        )

    def ref_for_tensor(self, tensor) -> BufferRef:
        """A BufferRef viewing a materialized ``Tensor`` arg over this handle.

        The view geometry (``byte_offset``, ``shapes``, ``strides``, ``dtype``) is read from the
        tensor; the backing is this handle. ``tensor.data`` must lie within this handle's backing.
        This is the per-arg step of submit-layer auto-wrap once the owning handle is chosen.
        """
        byte_offset = int(tensor.data) - self.base
        if byte_offset < 0 or byte_offset + int(tensor.nbytes()) > self.nbytes:
            raise ValueError(
                f"tensor view [{byte_offset}, {byte_offset + int(tensor.nbytes())}) lies outside "
                f"the handle's backing (nbytes={self.nbytes})"
            )
        return self.ref(
            shapes=tuple(tensor.shapes),
            strides=tuple(tensor.strides),
            dtype=int(tensor.dtype.value),
            byte_offset=byte_offset,
        )

    def close(self) -> None:
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None


def create_host_shared_buffer(
    nbytes: int,
    owner_instance_id: bytes,
    buffer_id: int,
    owner_worker_path: str = "",
    generation: int = 0,
    visibility: Visibility = Visibility.SHARED,
    access: AccessMode = AccessMode.READWRITE,
) -> BufferHandle:
    """Allocate a POSIX-shm host backing and wrap it as an owner ``BufferHandle`` (backend POSIX_SHM).

    The backend body is the shm name (UTF-8); the consumer maps it by name in ``ImportRegistry``.
    """
    if nbytes <= 0:
        raise ValueError(f"create_host_shared_buffer: nbytes must be positive, got {nbytes}")
    shm = SharedMemory(create=True, size=nbytes)
    identity = CanonicalIdentity(owner_instance_id, buffer_id, owner_worker_path, generation)
    return BufferHandle(
        identity=identity,
        address_space=AddressSpace.HOST,
        visibility=visibility,
        access=access,
        backend_kind=BackendKind.POSIX_SHM,
        nbytes=nbytes,
        body=shm.name.encode("utf-8"),
        shm=shm,
        base=_shm_base_addr(shm),
    )


def wrap_fork_inherited(
    data_ptr: int,
    nbytes: int,
    owner_instance_id: bytes,
    buffer_id: int,
    owner_worker_path: str = "",
    generation: int = 0,
    access: AccessMode = AccessMode.READ,
) -> BufferHandle:
    """Wrap a pre-fork, COW-inherited host allocation as a zero-copy ``FORK_SHM`` ``BufferHandle``.

    A tensor allocated before the chip children were forked is present in every child at the *same*
    virtual address (copy-on-write). The backend body is that base VA (u64 LE); the consumer
    materializes to the same VA with no mapping and no copy. Because COW splits a page on the child's
    first write, this is read-only from the child's side (``access`` defaults to READ) — an output the
    parent must read back needs a shm backing (``create_host_shared_buffer``) instead.
    """
    identity = CanonicalIdentity(owner_instance_id, buffer_id, owner_worker_path, generation)
    return BufferHandle(
        identity=identity,
        address_space=AddressSpace.HOST,
        visibility=Visibility.SHARED,
        access=access,
        backend_kind=BackendKind.FORK_SHM,
        nbytes=nbytes,
        body=int(data_ptr).to_bytes(8, "little"),
        shm=None,
        base=int(data_ptr),
    )


def wrap_posix_shm(
    shm_name: str,
    base: int,
    nbytes: int,
    owner_instance_id: bytes,
    buffer_id: int,
    owner_worker_path: str = "",
    generation: int = 0,
    access: AccessMode = AccessMode.READWRITE,
) -> BufferHandle:
    """Wrap an **already-existing** POSIX shm (owned elsewhere, e.g. a ``create_host_buffer``) as a
    ``POSIX_SHM`` ``BufferHandle`` carrying a canonical identity.

    The returned handle does NOT own the shm: ``shm`` is None so ``close()`` is a no-op, and the
    original allocator is responsible for unlinking it. Used by submit-layer auto-wrap to give an
    existing shared buffer a typed identity + descriptor without re-creating it.
    """
    identity = CanonicalIdentity(owner_instance_id, buffer_id, owner_worker_path, generation)
    return BufferHandle(
        identity=identity,
        address_space=AddressSpace.HOST,
        visibility=Visibility.SHARED,
        access=access,
        backend_kind=BackendKind.POSIX_SHM,
        nbytes=nbytes,
        body=shm_name.encode("utf-8"),
        shm=None,
        base=base,
    )


@dataclass
class ImportedBuffer:
    """A handle materialized into the consumer's address space: identity -> local base."""

    identity: CanonicalIdentity
    base: int
    nbytes: int
    address_space: AddressSpace = AddressSpace.HOST
    shm: SharedMemory | None = None  # the consumer's own mapping for shm backends


class ImportRegistry:
    """Per-consumer-endpoint lazy import cache: materialize a BufferRef's embedded descriptor to a
    local base on first receipt (map-once), keyed by canonical identity.

    A consumer calls ``materialize`` for each ref's embedded descriptor as it arrives; the first
    sight of an identity maps its backing into this process, later sights reuse the cached base
    (a bumped generation is a distinct identity, materialized fresh). Keyed by the packed canonical
    identity so lookups are exact — never a numeric-range guess.
    """

    def __init__(self) -> None:
        self._by_identity: dict[bytes, ImportedBuffer] = {}

    def materialize(self, descriptor: BufferHandleDescriptor | bytes) -> ImportedBuffer:
        """Map ``descriptor``'s backing into this process on first sight of its identity; reuse the
        cached ImportedBuffer thereafter (map-once)."""
        desc = BufferHandleDescriptor.unpack(descriptor) if isinstance(descriptor, (bytes, bytearray)) else descriptor
        key = desc.identity.pack()
        cached = self._by_identity.get(key)
        if cached is not None:
            return cached
        if desc.backend_kind == BackendKind.FORK_SHM:
            # COW-inherited at the same VA in every forked child: the body is the base VA (u64 LE),
            # already valid in this process — no mapping.
            base = int.from_bytes(desc.body, "little")
            imported = ImportedBuffer(desc.identity, base, desc.nbytes, desc.address_space, None)
        elif desc.backend_kind == BackendKind.POSIX_SHM:
            shm = SharedMemory(name=desc.body.decode("utf-8"))
            imported = ImportedBuffer(desc.identity, _shm_base_addr(shm), desc.nbytes, desc.address_space, shm)
        elif desc.backend_kind == BackendKind.REMOTE_SIDECAR:
            raise ValueError("ImportRegistry: REMOTE_SIDECAR backend is reserved for P2")
        else:
            raise NotImplementedError(f"ImportRegistry: backend {desc.backend_kind!r} not supported in P1-B")
        self._by_identity[key] = imported
        return imported

    def materialize_blob(self, blob_ptr: int, capacity: int) -> dict[bytes, tuple[int, int]]:
        """Lazily materialize every embedded descriptor in a BufferRef blob and return the resolved
        map for ``materialize_bufferref_blob``: packed identity -> (local base, address_space)."""
        for desc_bytes in bufferref_blob_descriptors(blob_ptr, capacity):
            self.materialize(desc_bytes)
        return self.materialization_map()

    def resolve(self, identity: CanonicalIdentity) -> ImportedBuffer:
        imported = self._by_identity.get(identity.pack())
        if imported is None:
            raise KeyError(f"ImportRegistry: no handle registered for {identity}")
        return imported

    def materialization_map(self) -> dict[bytes, tuple[int, int]]:
        """Snapshot for ``materialize_bufferref_blob``: packed identity -> (local base, address_space)."""
        return {key: (ib.base, int(ib.address_space)) for key, ib in self._by_identity.items()}

    def unregister(self, identity: CanonicalIdentity) -> None:
        imported = self._by_identity.pop(identity.pack(), None)
        if imported is not None and imported.shm is not None:
            imported.shm.close()

    def close(self) -> None:
        for imported in self._by_identity.values():
            if imported.shm is not None:
                imported.shm.close()
        self._by_identity.clear()
