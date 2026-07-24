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
from dataclasses import dataclass, replace
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
    DataType,
    bufferref_blob_descriptors,
    bufferref_blob_refs,
    get_element_size,
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


def _row_major_strides(shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Contiguous (row-major) element strides for ``shapes``: strides[i] = prod(shapes[i+1:])."""
    strides = [1] * len(shapes)
    for i in range(len(shapes) - 2, -1, -1):
        strides[i] = strides[i + 1] * shapes[i + 1]
    return tuple(strides)


@dataclass(frozen=True)
class BufferRef:
    """The blob-carried wire element: a full embedded handle descriptor + a strided view onto it.

    Self-describing — the consumer materializes ``handle`` on first receipt (no prior handshake),
    keyed by ``handle.identity``. Carries no materialized address.

    View algebra mirrors the C++ ``Tensor`` (``slice`` / ``transpose`` / ``permute`` / ``view`` /
    ``reshape``): pure metadata rewrites of ``(byte_offset, shapes, strides)`` returning a new
    ``BufferRef``. ``strides`` are ELEMENT strides (> 0); ``byte_offset`` is a BYTE offset. Matching
    ``Tensor``: ``slice`` / ``transpose`` / ``permute`` / ``view`` work on any (incl. strided) view;
    ``reshape`` requires a contiguous view (no allocating copy — reach contiguous first).
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

    @classmethod
    def unpack(cls, raw: bytes) -> BufferRef:
        handle = BufferHandleDescriptor.unpack(raw[:BUFFER_HANDLE_DESCRIPTOR_BYTES])
        vals = _BUFFER_REF_TAIL.unpack(raw[BUFFER_HANDLE_DESCRIPTOR_BYTES:BUFFER_REF_BYTES])
        byte_offset, ndims = vals[0], vals[1]
        shapes = tuple(vals[2 : 2 + ndims])
        strides = tuple(vals[2 + MAX_TENSOR_DIMS : 2 + MAX_TENSOR_DIMS + ndims])
        dtype = vals[2 + 2 * MAX_TENSOR_DIMS]
        return cls(handle=handle, byte_offset=byte_offset, shapes=shapes, strides=strides, dtype=dtype)

    # --- view algebra (zero-copy metadata rewrites, mirroring Tensor) ------------------------------

    @property
    def ndims(self) -> int:
        return len(self.shapes)

    def numel(self) -> int:
        n = 1
        for s in self.shapes:
            n *= s
        return n

    @property
    def is_contiguous(self) -> bool:
        return self.strides == _row_major_strides(self.shapes)

    def _elem_bytes(self) -> int:
        return get_element_size(DataType(self.dtype))

    def slice(self, dim: int, start: int, end: int, step: int = 1) -> BufferRef:
        """View ``[start, end)`` with positive ``step`` along ``dim``. Works on any (strided) view."""
        if not 0 <= dim < self.ndims:
            raise ValueError(f"slice dim {dim} out of range [0, {self.ndims})")
        if step < 1:
            raise ValueError(f"slice step must be >= 1, got {step}")
        if not 0 <= start < end <= self.shapes[dim]:
            raise ValueError(f"slice [{start}, {end}) out of range [0, {self.shapes[dim]}] on dim {dim}")
        old_stride = self.strides[dim]
        new_shapes = list(self.shapes)
        new_strides = list(self.strides)
        new_shapes[dim] = (end - start + step - 1) // step
        new_strides[dim] = old_stride * step
        byte_offset = self.byte_offset + start * old_stride * self._elem_bytes()
        return replace(self, byte_offset=byte_offset, shapes=tuple(new_shapes), strides=tuple(new_strides))

    def transpose(self, x: int, y: int) -> BufferRef:
        """Swap dims ``x`` and ``y`` (shapes + strides). Works on any (strided) view."""
        if not (0 <= x < self.ndims and 0 <= y < self.ndims):
            raise ValueError(f"transpose dims ({x}, {y}) out of range [0, {self.ndims})")
        new_shapes = list(self.shapes)
        new_strides = list(self.strides)
        new_shapes[x], new_shapes[y] = new_shapes[y], new_shapes[x]
        new_strides[x], new_strides[y] = new_strides[y], new_strides[x]
        return replace(self, shapes=tuple(new_shapes), strides=tuple(new_strides))

    def permute(self, order: tuple[int, ...]) -> BufferRef:
        """Reorder dims by ``order`` (a permutation of range(ndims)). Works on any (strided) view."""
        if sorted(order) != list(range(self.ndims)):
            raise ValueError(f"permute order {order} must be a permutation of range({self.ndims})")
        new_shapes = tuple(self.shapes[o] for o in order)
        new_strides = tuple(self.strides[o] for o in order)
        return replace(self, shapes=new_shapes, strides=new_strides)

    def view(self, view_shapes: tuple[int, ...], view_offsets: tuple[int, ...]) -> BufferRef:
        """Sub-view: origin shifts by ``view_offsets`` (per dim), shape becomes ``view_shapes``,
        strides unchanged. Each ``view_offsets[i] + view_shapes[i]`` must stay within ``shapes[i]``.
        """
        if len(view_shapes) != self.ndims or len(view_offsets) != self.ndims:
            raise ValueError(f"view shapes/offsets must have ndims={self.ndims}")
        elem = self._elem_bytes()
        byte_offset = self.byte_offset
        for i in range(self.ndims):
            if view_offsets[i] + view_shapes[i] > self.shapes[i]:
                raise ValueError(
                    f"view dim {i}: offset {view_offsets[i]} + shape {view_shapes[i]} exceeds {self.shapes[i]}"
                )
            byte_offset += view_offsets[i] * self.strides[i] * elem
        return replace(self, byte_offset=byte_offset, shapes=tuple(view_shapes))

    def reshape(self, new_shapes: tuple[int, ...]) -> BufferRef:
        """Contiguous-only reshape (mirrors Tensor::reshape's ``always_assert(is_contiguous)``)."""
        if not self.is_contiguous:
            raise ValueError("reshape requires a contiguous view; reach contiguous (via a copy) first")
        n = 1
        for s in new_shapes:
            n *= s
        if n != self.numel():
            raise ValueError(f"reshape {new_shapes} (numel {n}) does not match current numel {self.numel()}")
        return replace(self, shapes=tuple(new_shapes), strides=_row_major_strides(tuple(new_shapes)))


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
        dtype: int,
        strides: tuple[int, ...] | None = None,
        byte_offset: int = 0,
    ) -> BufferRef:
        """A self-describing BufferRef viewing this handle: embeds the full descriptor + the view.

        ``strides`` default to contiguous (row-major) — ``handle.ref(shape, dtype)`` names the whole
        buffer as a contiguous view; pass explicit element strides for a strided view (or reach one
        via the BufferRef view algebra). ``byte_offset`` must be a multiple of the dtype size (checked
        at materialization).
        """
        shapes = tuple(shapes)
        strides = _row_major_strides(shapes) if strides is None else tuple(strides)
        return BufferRef(
            handle=self.to_descriptor(),
            byte_offset=byte_offset,
            shapes=shapes,
            strides=strides,
            dtype=int(dtype),
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


def re_export(
    source: BufferHandleDescriptor,
    owner_instance_id: bytes,
    buffer_id: int,
    owner_worker_path: str = "",
    generation: int = 1,
) -> BufferHandle:
    """Re-export a received handle descriptor under a NEW local identity, WITHOUT mapping.

    Each L3+ level's orch sees only its own handles: a ref to an upper-level backing is re-exported to
    a local handle ``H'`` on receipt. ``H'`` carries a fresh identity (this level owns it) but the
    SAME backing as ``source`` (backend_kind / body / nbytes / address_space / visibility / access).
    No mmap — ``base=0``, ``shm=None``; a downstream consumer materializes the backing lazily (only a
    compute leaf actually maps). Re-export is per-backing (memoize by ``source.identity``), so pure
    forwarding carries no per-ref map cost.
    """
    identity = CanonicalIdentity(owner_instance_id, buffer_id, owner_worker_path, generation)
    return BufferHandle(
        identity=identity,
        address_space=source.address_space,
        visibility=source.visibility,
        access=source.access,
        backend_kind=source.backend_kind,
        nbytes=source.nbytes,
        body=source.body,
        shm=None,
        base=0,
    )


def remote_sidecar_ref(
    shapes: tuple[int, ...],
    dtype: int,
    nbytes: int,
    owner_worker_id: int,
    buffer_id: int,
    generation: int,
    address_space: AddressSpace,
) -> BufferRef:
    """Build a ``REMOTE_SIDECAR`` BufferRef for a task arg destined for a remote worker.

    An arg passed L4→remote-L3 cannot be materialized from a local backing — the data lives on another
    machine and travels via the remote transport. Its BufferRef therefore carries ``backend_kind =
    REMOTE_SIDECAR`` (a consumer decode-rejects a local materialize; the authoritative remote
    descriptor rides in the per-task RemoteTaskArgsSidecar). The identity encodes the remote buffer
    (``owner_worker_id`` folded into the opaque nonce, plus ``buffer_id`` / ``generation``) so
    dependency inference and routing stay stable across the hop.
    """
    oid = int(owner_worker_id).to_bytes(OWNER_INSTANCE_ID_BYTES, "little")
    identity = CanonicalIdentity(oid, buffer_id, f"remote/{owner_worker_id}", generation)
    handle = BufferHandleDescriptor(
        identity=identity,
        address_space=address_space,
        visibility=Visibility.SHARED,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.REMOTE_SIDECAR,
        nbytes=nbytes,
        body=b"",
    )
    shapes = tuple(shapes)
    return BufferRef(
        handle=handle,
        byte_offset=0,
        shapes=shapes,
        strides=_row_major_strides(shapes),
        dtype=int(dtype),
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


def wrap_device_malloc(
    device_ptr: int,
    nbytes: int,
    owner_instance_id: bytes,
    buffer_id: int,
    owner_worker_path: str = "",
    generation: int = 0,
    access: AccessMode = AccessMode.READWRITE,
) -> BufferHandle:
    """Wrap a device pointer (from ``orch.malloc``) as a ``DEVICE_MALLOC`` ``BufferHandle``.

    The backend body is the device pointer (u64 LE); the consumer materializes to that pointer with no
    mapping. The pointer is valid only on the chip that allocated it, so a ref over this handle must be
    dispatched only to that chip (a topology invariant, as for the former ``child_memory`` tensor).
    """
    identity = CanonicalIdentity(owner_instance_id, buffer_id, owner_worker_path, generation)
    return BufferHandle(
        identity=identity,
        address_space=AddressSpace.DEVICE,
        visibility=Visibility.PRIVATE,
        access=access,
        backend_kind=BackendKind.DEVICE_MALLOC,
        nbytes=nbytes,
        body=int(device_ptr).to_bytes(8, "little"),
        shm=None,
        base=int(device_ptr),
    )


@dataclass
class ImportedBuffer:
    """A handle materialized into the consumer's address space: identity -> local base."""

    identity: CanonicalIdentity
    base: int
    nbytes: int
    address_space: AddressSpace = AddressSpace.HOST
    shm: SharedMemory | None = None  # the consumer's own mapping for shm backends


@dataclass
class MappedArg:
    """A Python compute (sub-worker) task arg: a BufferRef materialized into this process, exposing a
    writable ``buffer`` at the view origin plus the view geometry. The callable computes with e.g.
    ``torch.frombuffer(arg.buffer, dtype=<from arg.dtype>, count=prod(arg.shapes))`` — reads/writes
    land in the shared backing the owner sees.
    """

    imported: ImportedBuffer
    byte_offset: int
    shapes: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: int  # DataType value

    @property
    def buffer(self) -> memoryview:
        """A memoryview over the mapped backing at this view's origin (``byte_offset``)."""
        ib = self.imported
        if ib.shm is not None:
            base = ib.shm.buf
            assert base is not None
        else:
            # FORK_SHM (COW): no shm object — wrap the inherited VA range.
            base = memoryview((ctypes.c_char * ib.nbytes).from_address(ib.base))
        return base[self.byte_offset :]


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
        if desc.backend_kind in (BackendKind.FORK_SHM, BackendKind.DEVICE_MALLOC):
            # The body is the base pointer (u64 LE), already valid in this process — no mapping.
            # FORK_SHM: a COW-inherited host VA. DEVICE_MALLOC: a device pointer valid on the chip that
            # allocated it (the ref must only reach that chip — a topology invariant).
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

    def mapped_args_from_blob(self, blob_ptr: int, capacity: int) -> list[MappedArg]:
        """Materialize every ref in a BufferRef blob into a MappedArg for a Python compute callable:
        map each backing (map-once) and expose a buffer at the view origin. This is the compute-leaf
        map (a sub-worker reads/writes), distinct from pure forwarding (re-export, which never maps).
        """
        return [
            MappedArg(self.materialize(ref.handle), ref.byte_offset, ref.shapes, ref.strides, ref.dtype)
            for ref in (BufferRef.unpack(rb) for rb in bufferref_blob_refs(blob_ptr, capacity))
        ]

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
