# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Owner/consumer side of the Buffer ABI.

The three wire types — ``CanonicalIdentity``, ``BufferDescriptor`` and ``Tensor`` — are the C++
structs of ``buffer.h`` bound directly, so there is one definition of the layout and one validator
(``validate_tensor``) behind every boundary that builds or decodes one. This module re-exports them
and adds what is inherently host-side: ``Buffer``, which owns a POSIX shm, the constructors that wrap
a backing as one, and ``ImportRegistry``, the consumer's map-once cache. Transporting a ``Tensor``
is the TaskArgs mailbox blob's job and lives in ``task_args.h``, not here.

``CanonicalIdentity`` is fixed-length (opaque ``owner_instance_id`` + ``buffer_id`` + ``generation``)
so hashing and comparison cannot read past it whatever arrives on the wire. The owning worker's tree
path is **not** part of the identity: it is interned to a diagnostic ``owner_worker_path_id`` whose
side table lives only in the owning process.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from multiprocessing.shared_memory import SharedMemory
from typing import Any

from _task_interface import (  # pyright: ignore[reportMissingImports]
    OWNER_INSTANCE_ID_BYTES,
    AccessMode,
    AddressSpace,
    BackendKind,
    BufferDescriptor,
    CanonicalIdentity,
    DataType,
    Tensor,
    read_args_from_blob,
)

from .comm_endpoints import AdapterKind, AdapterProfile, RegionAccessReasonCode

__all__ = [
    "AccessMode",
    "AddressSpace",
    "BackendKind",
    "Buffer",
    "BufferCapability",
    "BufferDescriptor",
    "CanonicalIdentity",
    "ImportContext",
    "ImportRegistry",
    "ImportedBuffer",
    "MappedArg",
    "MappedArgs",
    "Tensor",
    "capabilities_for_adapter",
    "create_host_shared_buffer",
    "host_ptr_nbytes",
    "intern_worker_path",
    "mint_owner_instance_id",
    "re_export",
    "remote_backing_identity",
    "remote_sidecar_tensor",
    "worker_path_for_id",
    "wrap_device_malloc",
    "wrap_fork_inherited",
    "wrap_vmm_window",
]


# Owner-side residency for the diagnostic worker path. Ids are process-local and meaningful only in
# the owning process: a consumer that cannot resolve one renders it as "<path#N>". Nothing routes,
# gates or keys on a path, so an unresolvable id is never an error. Id 0 means "no path".
_PATH_BY_ID: dict[int, str] = {0: ""}
_ID_BY_PATH: dict[str, int] = {"": 0}


def intern_worker_path(path: str) -> int:
    """The diagnostic id for ``path`` in this process, assigning one on first sight."""
    pid = _ID_BY_PATH.get(path)
    if pid is None:
        pid = len(_ID_BY_PATH)
        _ID_BY_PATH[path] = pid
        _PATH_BY_ID[pid] = path
    return pid


def worker_path_for_id(path_id: int) -> str:
    """``path_id`` rendered for humans; an id minted in another process has no local text."""
    return _PATH_BY_ID.get(int(path_id), f"<path#{int(path_id)}>")


# `owner_worker_path` is resolved through this module's intern table, which is process-local, so it
# hangs off the bound descriptor here rather than in the binding.
BufferDescriptor.owner_worker_path = property(
    lambda self: worker_path_for_id(self.owner_worker_path_id),
    doc="The owning worker's tree path, for diagnostics only; ``<path#N>`` when minted elsewhere.",
)


def _row_major_strides(shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Contiguous (row-major) element strides for ``shapes``: strides[i] = prod(shapes[i+1:])."""
    strides = [1] * len(shapes)
    for i in range(len(shapes) - 2, -1, -1):
        strides[i] = strides[i + 1] * shapes[i + 1]
    return tuple(strides)


def mint_owner_instance_id() -> bytes:
    """A fresh opaque nonce, unique per owner incarnation (defends identity against ABA).

    Must stay a full-width random draw. It is the only thing separating two Workers' buffer_id spaces,
    so a structured value (timestamp/pid) would hand the same identity to two Workers constructed in
    one process within one second — a routine pattern (an L4 and its L3 built back to back).
    """
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
class Buffer:
    """Owner-side registry object for one shared backing; owns the POSIX shm that backs it."""

    identity: CanonicalIdentity
    address_space: AddressSpace
    access: AccessMode
    backend_kind: BackendKind
    nbytes: int
    body: bytes = b""
    owner_worker_path_id: int = 0
    shm: SharedMemory | None = None
    base: int = 0
    # Owner-side only, never serialized into the descriptor: which next-level worker a DEVICE_MALLOC
    # backing lives on (0 for a host backing or an L2 own-device malloc). The device-pointer provenance
    # guard and free/copy key on (owner_worker_id, base).
    owner_worker_id: int = 0
    closed: bool = False
    # Owner-side only: whether the backing's name has actually been removed. Distinct from `closed`,
    # which is the derivation gate — a close() whose unlink raised leaves this false so a retry
    # attempts the unlink again.
    unlinked: bool = False

    def to_descriptor(self) -> BufferDescriptor:
        """The wire descriptor for this backing — what a consumer needs to resolve it."""
        if self.closed:
            raise ValueError(f"Buffer: cannot derive a descriptor from a released buffer ({self.identity})")
        return BufferDescriptor(
            identity=self.identity,
            address_space=self.address_space,
            owner_worker_path_id=self.owner_worker_path_id,
            access=self.access,
            backend_kind=self.backend_kind,
            nbytes=self.nbytes,
            body=self.body,
        )

    def tensor(
        self,
        shapes: tuple[int, ...],
        dtype: int | DataType,
        strides: tuple[int, ...] | None = None,
        byte_offset: int = 0,
    ) -> Tensor:
        """A self-describing ``Tensor`` viewing this buffer: embeds the full descriptor + the view.

        ``strides`` default to contiguous (row-major) — ``buffer.tensor(shape, dtype)`` names the
        whole buffer as a contiguous view; pass explicit element strides for a strided view.
        ``byte_offset`` must be a multiple of the dtype size (checked at materialization).
        ``dtype`` accepts a ``DataType`` enum or its int value.
        """
        shapes = tuple(shapes)
        strides = _row_major_strides(shapes) if strides is None else tuple(strides)
        return Tensor(
            buffer=self.to_descriptor(),
            byte_offset=byte_offset,
            shapes=shapes,
            strides=strides,
            dtype=dtype,
        )

    def close(self) -> None:
        """Release the backing. The owner unlinks it, so a later consumer map fails rather than
        resolving a name whose bytes are gone. Idempotent; a released Buffer's ``tensor()``/
        ``to_descriptor()`` are refused rather than building a view over memory that may already be
        gone — that refusal holds from the first call on, whether or not the release itself
        succeeded, since a partly-released backing is no safer to hand out than a fully released one.

        Each OS action succeeds at most once and is retried until it does. ``shm.close()`` runs
        first and never gates the unlink: the named backing outlives this process, so it is the leak
        worth removing even when the local unmap raised. An unlink that raises leaves ``shm`` in
        place, so a second ``close()`` attempts it again — that retry is what
        ``Worker._release_all_buffers`` leaves the registry entry behind for.
        """
        self.closed = True
        shm = self.shm
        if shm is None:
            return
        try:
            shm.close()
        finally:
            if not self.unlinked:
                shm.unlink()
                self.unlinked = True
        self.shm = None


def create_host_shared_buffer(
    nbytes: int,
    owner_instance_id: bytes,
    buffer_id: int,
    owner_worker_path: str = "",
    generation: int = 1,
    access: AccessMode = AccessMode.READWRITE,
) -> Buffer:
    """Allocate a POSIX-shm host backing and wrap it as an owner ``Buffer`` (backend POSIX_SHM).

    The backend body is the shm name (UTF-8); the consumer maps it by name in ``ImportRegistry``.
    """
    if nbytes <= 0:
        raise ValueError(f"create_host_shared_buffer: nbytes must be positive, got {nbytes}")
    shm = SharedMemory(create=True, size=nbytes)
    identity = CanonicalIdentity(owner_instance_id, buffer_id, generation)
    return Buffer(
        identity=identity,
        owner_worker_path_id=intern_worker_path(owner_worker_path),
        address_space=AddressSpace.HOST,
        access=access,
        backend_kind=BackendKind.POSIX_SHM,
        nbytes=nbytes,
        body=shm.name.encode("utf-8"),
        shm=shm,
        base=_shm_base_addr(shm),
    )


def re_export(source: BufferDescriptor) -> Buffer:
    """Re-export a received buffer descriptor for forwarding — identity **invariant**, no mapping.

    Canonical identity is invariant across every edge (frozen model §5/§8): the re-exported ``H'``
    keeps the SOURCE ``(owner_instance_id, buffer_id, generation)`` and the SAME
    backing (backend_kind / body / nbytes / address_space / access) as ``source`` — an
    L4-owned buffer forwarded L4→L3→L2 carries one identity at all three layers. Only the mapping is
    stripped: ``base=0``, ``shm=None`` (no mmap on the forwarding hop); a downstream compute leaf
    materializes lazily. Dependency inference keys on the (invariant) identity, so an alias /
    retain-release does not split across layers. Re-export is per-backing (memoize by identity), so
    pure forwarding carries no per-tensor map cost.
    """
    return Buffer(
        identity=source.identity,
        owner_worker_path_id=source.owner_worker_path_id,
        address_space=source.address_space,
        access=source.access,
        backend_kind=source.backend_kind,
        nbytes=source.nbytes,
        body=source.body,
        shm=None,
        base=0,
    )


def remote_backing_identity(owner_worker_id: int, buffer_id: int, generation: int) -> CanonicalIdentity:
    """The canonical identity of a backing that lives on another machine's worker.

    A remote owner's ``owner_instance_id`` never crosses the remote L3 wire, so the nonce is the
    owning worker's id instead. This is the single rule for naming a remote backing: the submitting
    L4's ``REMOTE_SIDECAR`` placeholder and the importing session runner both derive from it, so one
    remote backing carries one identity on both sides of the hop.
    """
    oid = int(owner_worker_id).to_bytes(OWNER_INSTANCE_ID_BYTES, "little")
    # A HOST_INLINE placeholder has no backing and so no generation of its own; 0 is the reserved
    # "uninitialized" value a decoder rejects, so it carries the initial generation instead.
    return CanonicalIdentity(oid, int(buffer_id), int(generation) or 1)


def remote_sidecar_tensor(
    shapes: tuple[int, ...],
    dtype: int,
    nbytes: int,
    owner_worker_id: int,
    buffer_id: int,
    generation: int,
    address_space: AddressSpace,
    byte_offset: int = 0,
) -> Tensor:
    """Build a ``REMOTE_SIDECAR`` ``Tensor`` for a task arg destined for a remote worker.

    An arg passed L4→remote-L3 cannot be materialized from a local backing — the data lives on another
    machine and travels via the remote transport. Its descriptor therefore carries ``backend_kind =
    REMOTE_SIDECAR`` (a consumer decode-rejects a local materialize; the authoritative remote
    descriptor rides in the per-task RemoteTaskArgsSidecar). The identity encodes the remote buffer
    (``owner_worker_id`` folded into the opaque nonce, plus ``buffer_id`` / ``generation``) so
    dependency inference and routing stay stable across the hop.

    ``nbytes`` is the whole backing length and ``byte_offset`` is the view origin, matching the
    ordinary local Tensor ABI. The view's transport range is carried separately in the sidecar.
    This placeholder is what the remote L3 wire carries verbatim as the task's per-argument record.
    """
    descriptor = BufferDescriptor(
        identity=remote_backing_identity(owner_worker_id, buffer_id, generation),
        owner_worker_path_id=intern_worker_path(f"remote/{owner_worker_id}"),
        address_space=address_space,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.REMOTE_SIDECAR,
        # The placeholder follows the normal Tensor ABI: this is the whole backing length. The
        # remote sidecar carries the transport view range separately.
        nbytes=nbytes,
        body=b"",
    )
    shapes = tuple(shapes)
    return Tensor(
        buffer=descriptor,
        byte_offset=byte_offset,
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
    generation: int = 1,
    access: AccessMode = AccessMode.READ,
    backend_kind: BackendKind = BackendKind.FORK_COW,
) -> Buffer:
    """Wrap a pre-fork, fork-inherited host allocation as a zero-copy ``Buffer``.

    Memory allocated before the children were forked is present in every child at the *same* virtual
    address; the backend body is that base VA (u64 LE) and the consumer materializes to the same VA
    with no mapping and no copy. ``backend_kind`` states which mmap the caller actually holds, and is
    a classification the caller must make rather than something inferred from ``access``:

    * ``FORK_SHM`` — ``MAP_SHARED`` (e.g. a ``torch.Tensor.share_memory_()``): a child's writes land
      in the pages the parent reads, so it can serve as an OUTPUT. Any ``access`` is legal.
    * ``FORK_COW`` — plain ``MAP_PRIVATE``: copy-on-write, so a child's first write splits the page
      into a private copy the parent never sees. ``access`` must be ``READ``; the descriptor's
      validator rejects anything else.

    The two are not interchangeable and neither implies an ``access``: a ``MAP_SHARED`` backing
    granted READ only is a legal, expressible combination.
    """
    identity = CanonicalIdentity(owner_instance_id, buffer_id, generation)
    return Buffer(
        identity=identity,
        owner_worker_path_id=intern_worker_path(owner_worker_path),
        address_space=AddressSpace.HOST,
        access=access,
        backend_kind=backend_kind,
        nbytes=nbytes,
        body=int(data_ptr).to_bytes(8, "little"),
        shm=None,
        base=int(data_ptr),
    )


def host_ptr_nbytes(obj: Any) -> tuple[int, int]:
    """Host address + byte length of a copy_to/copy_from buffer, without importing torch.

    A torch tensor is read via its ``data_ptr`` / ``numel`` / ``element_size`` (duck-typed); any other
    object goes through the buffer protocol and must be writable so its backing address is stable for
    the duration of the synchronous copy.
    """
    if hasattr(obj, "data_ptr") and hasattr(obj, "numel") and hasattr(obj, "element_size"):
        return int(obj.data_ptr()), int(obj.numel()) * int(obj.element_size())
    mv = memoryview(obj)
    if mv.readonly:
        raise TypeError("copy_to/copy_from host buffer must be a torch tensor or a writable buffer")
    return ctypes.addressof((ctypes.c_char * mv.nbytes).from_buffer(obj)), mv.nbytes


def wrap_device_malloc(
    device_ptr: int,
    nbytes: int,
    owner_instance_id: bytes,
    buffer_id: int,
    owner_worker_path: str = "",
    generation: int = 1,
    access: AccessMode = AccessMode.READWRITE,
    owner_worker_id: int = 0,
) -> Buffer:
    """Wrap a device pointer (from a worker device malloc) as a ``DEVICE_MALLOC`` ``Buffer``.

    The backend body is the device pointer (u64 LE); the consumer materializes to that pointer with no
    mapping. The pointer is valid only on the chip that allocated it, so a tensor over this buffer must be
    dispatched only to that chip (a topology invariant, as for the former ``child_memory`` tensor).
    ``owner_worker_id`` records which next-level worker the backing lives on for free/copy provenance.
    """
    identity = CanonicalIdentity(owner_instance_id, buffer_id, generation)
    return Buffer(
        identity=identity,
        owner_worker_path_id=intern_worker_path(owner_worker_path),
        address_space=AddressSpace.DEVICE,
        access=access,
        backend_kind=BackendKind.DEVICE_MALLOC,
        nbytes=nbytes,
        body=int(device_ptr).to_bytes(8, "little"),
        shm=None,
        base=int(device_ptr),
        owner_worker_id=int(owner_worker_id),
    )


def wrap_vmm_window(
    device_ptr: int,
    nbytes: int,
    owner_instance_id: bytes,
    buffer_id: int,
    owner_worker_path: str = "",
    generation: int = 1,
    access: AccessMode = AccessMode.READWRITE,
    owner_worker_id: int = 0,
) -> Buffer:
    """Wrap a domain-window-carved device VA as a ``VMM_WINDOW`` ``Buffer``.

    A comm domain's per-rank window is device memory carved by ``allocate_domain``; each named buffer
    slice is one such backing. The backend body is the device VA (u64 LE); the consumer materializes to
    that VA with no mapping. The VA is valid only on the chip that owns the window, so a tensor over this
    buffer must be dispatched only to that chip (``owner_worker_id``). Unlike ``DEVICE_MALLOC`` it is
    not freed by ``worker.free`` — the domain owns its lifetime and reclaims it at ``release_domain``.
    """
    identity = CanonicalIdentity(owner_instance_id, buffer_id, generation)
    return Buffer(
        identity=identity,
        owner_worker_path_id=intern_worker_path(owner_worker_path),
        address_space=AddressSpace.DEVICE,
        access=access,
        backend_kind=BackendKind.VMM_WINDOW,
        nbytes=nbytes,
        body=int(device_ptr).to_bytes(8, "little"),
        shm=None,
        base=int(device_ptr),
        owner_worker_id=int(owner_worker_id),
    )


def _descriptor_delta(a: BufferDescriptor, b: BufferDescriptor) -> str:
    """The fields on which two descriptors differ, as ``name: old -> new``.

    A whole ``repr`` of both would carry a 32-byte body twice for what is usually a one-field
    disagreement, and the reader still has to diff them by eye.
    """
    fields = (
        ("nbytes", a.nbytes, b.nbytes),
        ("access", a.access, b.access),
        ("backend_kind", a.backend_kind, b.backend_kind),
        ("address_space", a.address_space, b.address_space),
        ("owner_worker_path_id", a.owner_worker_path_id, b.owner_worker_path_id),
        ("body", a.body, b.body),
    )
    diff = [f"{name}: {old!r} -> {new!r}" for name, old, new in fields if old != new]
    return ", ".join(diff) if diff else "no field differs"


class BufferCapability(str, Enum):
    """An operation an endpoint is allowed to perform on an attached backing.

    This is intentionally distinct from :class:`~simpler.comm_endpoints.AdapterKind`: the latter
    names the mechanism used to attach, while this type records the resulting access rights.  One
    adapter can afford several rights and the same right can be afforded by several mechanisms.

    Load and store are separate rights because a backing can afford one without the other: a
    ``FORK_COW`` backing is dereferenceable in the child, yet a store there splits into a private
    copy the owner never sees, which is why its descriptor carries ``AccessMode.READ``.
    """

    DIRECT_LOAD = "DIRECT_LOAD"
    DIRECT_STORE = "DIRECT_STORE"
    COPY_TO = "COPY_TO"
    COPY_FROM = "COPY_FROM"
    DEVICE_PEER_ACCESS = "DEVICE_PEER_ACCESS"


# The rights that write to the backing, and the ones that read it. `AccessMode` withholds one
# group or the other, so every capability belongs to exactly one of them (`DEVICE_PEER_ACCESS`
# names a device-side attachment rather than a direction, so it survives either restriction).
_WRITE_CAPABILITIES = frozenset({BufferCapability.DIRECT_STORE, BufferCapability.COPY_TO})
_READ_CAPABILITIES = frozenset({BufferCapability.DIRECT_LOAD, BufferCapability.COPY_FROM})

_CAPABILITIES_BY_ADAPTER: dict[AdapterKind, frozenset[BufferCapability]] = {
    AdapterKind.DIRECT_MAP: frozenset(
        {
            BufferCapability.DIRECT_LOAD,
            BufferCapability.DIRECT_STORE,
            BufferCapability.COPY_TO,
            BufferCapability.COPY_FROM,
        }
    ),
    AdapterKind.DEVICE_PEER: frozenset(
        {
            BufferCapability.DEVICE_PEER_ACCESS,
            BufferCapability.COPY_TO,
            BufferCapability.COPY_FROM,
        }
    ),
    AdapterKind.OWNER_DELEGATED_COPY: frozenset({BufferCapability.COPY_TO, BufferCapability.COPY_FROM}),
    # These mechanisms create/use a different backing or describe an operation, so they do not
    # grant an attachment capability on the source backing.
    AdapterKind.EXPLICIT_TRANSFER: frozenset(),
    AdapterKind.COLLECTIVE: frozenset(),
}


def capabilities_for_adapter(
    kind: AdapterKind | str, access: AccessMode | int | None = None
) -> frozenset[BufferCapability]:
    """Return the access rights afforded by ``kind``, narrowed to what ``access`` permits.

    The mechanism sets the ceiling and the backing's own ``AccessMode`` lowers it: a mechanism
    that could dereference both ways still grants no write right over a READ backing.  Passing no
    ``access`` asks for the mechanism's ceiling alone.

    Unknown values fail closed with an empty set.  That is important at this boundary: a new
    adapter must not silently acquire direct access until its capability mapping is deliberate.
    """

    try:
        normalized = AdapterKind(kind)
    except (TypeError, ValueError):
        return frozenset()
    granted = _CAPABILITIES_BY_ADAPTER.get(normalized, frozenset())
    if access is None:
        return granted
    try:
        mode = AccessMode(access)
    except (TypeError, ValueError):
        return frozenset()
    if mode is AccessMode.READ:
        return granted - _WRITE_CAPABILITIES
    if mode is AccessMode.WRITE:
        return granted - _READ_CAPABILITIES
    return granted


# The consumer-side mechanism each wire backend resolves through in this process. Two backends
# share a mechanism wherever they resolve the same way and differ only in what the descriptor says
# about them: FORK_SHM/FORK_COW are both a VA inherited across the fork (they differ in `access`),
# and DEVICE_MALLOC/VMM_WINDOW are both a VA valid only on the chip that owns them.
_PROFILE_BY_BACKEND: dict[BackendKind, AdapterProfile] = {
    BackendKind.FORK_SHM: AdapterProfile.FORK_INHERITED_VA,
    BackendKind.FORK_COW: AdapterProfile.FORK_INHERITED_VA,
    BackendKind.POSIX_SHM: AdapterProfile.HOST_SHM_MAP,
    BackendKind.DEVICE_MALLOC: AdapterProfile.DEVICE_LOCAL,
    BackendKind.VMM_WINDOW: AdapterProfile.DEVICE_LOCAL,
}

# How a host endpoint reaches a chip-owned device backing it cannot map: by asking the chip that
# owns it to perform the copy. Keyed on the backend because that is what the two moments of this
# vocabulary have in common -- region planning already names the `VMM_WINDOW` half `HOST_VMM_COPY`
# for a host region consumer, and never sees a `DEVICE_MALLOC` at all.
_DELEGATED_PROFILE_BY_BACKEND: dict[BackendKind, AdapterProfile] = {
    BackendKind.DEVICE_MALLOC: AdapterProfile.OWNER_DEVICE_COPY,
    BackendKind.VMM_WINDOW: AdapterProfile.HOST_VMM_COPY,
}


def select_adapter(desc: BufferDescriptor, context: ImportContext) -> tuple[AdapterKind, AdapterProfile]:
    """The mechanism ``context``'s endpoint reaches ``desc``'s backing by -- mapping or not.

    The capability judgment at its per-tensor evaluation moment, and the counterpart of the region
    planner's ``_adapter_candidates``. Two of the four axes that planner ranges over are bound to
    constants here rather than absent: connectivity is fixed local because the consumer *is* this
    process and transport has already happened, and the consumer's deployment arrived as the
    ``ImportContext`` the owner handed down at fork.

    **Not being able to map a backing is not the same as not being allowed to touch it.** A host
    endpoint cannot hold a device VA, but the Worker that owns the chip can still reach that
    backing by asking the chip to copy -- which is what ``Worker.copy_to`` / ``copy_from`` do. So a
    device backing at a host endpoint resolves to ``OWNER_DELEGATED_COPY`` when this endpoint has a
    delegation channel to its owner, and is refused only when it has none (a SUB child, or another
    Worker's backing entirely). ``materialize`` is what narrows this to the mapping subset, because
    a local base is what *it* has to return -- the narrowing belongs there and not here.

    The owner-nonce check is Worker-grained, not chip-grained (see ``ImportContext``): it cannot by
    itself tell apart two chips forked from the same multi-device Worker. The exact-chip half is
    the dispatch guard's (``Worker._child_prov_check_dispatch``, which compares the registered
    handle's ``owner_worker_id``); this is the backstop for a path that arrives without one.
    """
    if desc.backend_kind == BackendKind.REMOTE_SIDECAR:
        raise _refuse(
            RegionAccessReasonCode.UNSUPPORTED_BACKEND_KIND,
            f"REMOTE_SIDECAR ({desc.identity}) names a backing on another machine; the remote "
            f"session resolves it against its own registry, so it is never reached from here",
        )
    if desc.address_space == AddressSpace.DEVICE:
        # One nonce comparison for both endpoint kinds; `is_host_endpoint` then says whether being
        # related to that Worker means being one of its chips or owning them.
        if context.device_owner_instance_id != bytes(desc.identity.owner_instance_id):
            raise _refuse(
                RegionAccessReasonCode.UNSUPPORTED_ENDPOINT_RELATION,
                f"DEVICE backing ({desc.identity}) belongs to a Worker this endpoint is unrelated "
                f"to: it is neither one of that Worker's chips nor the owner of them",
            )
        if context.is_host_endpoint:
            delegated = _DELEGATED_PROFILE_BY_BACKEND.get(desc.backend_kind)
            if delegated is None:
                raise _refuse(
                    RegionAccessReasonCode.UNSUPPORTED_BACKEND_KIND,
                    f"no delegated-copy mechanism for backend {desc.backend_kind!r} ({desc.identity})",
                )
            return AdapterKind.OWNER_DELEGATED_COPY, delegated
    profile = _PROFILE_BY_BACKEND.get(desc.backend_kind)
    if profile is None:
        raise _refuse(
            RegionAccessReasonCode.UNSUPPORTED_BACKEND_KIND,
            f"no local materialization mechanism for backend {desc.backend_kind!r} ({desc.identity})",
        )
    return AdapterKind.DIRECT_MAP, profile


def _refuse(code: RegionAccessReasonCode, message: str) -> ValueError:
    """A materialization refusal tagged with the reason vocabulary the region planner reports.

    Every refusal on this path carries one, so a caller can tell an endpoint-relation verdict from
    an unsupported backing without parsing prose, and so both evaluation moments of the capability
    judgment answer in the same terms.
    """
    return ValueError(f"ImportRegistry: [{code.value}] {message}")


def _resolve_body_as_va(desc: BufferDescriptor) -> tuple[int, SharedMemory | None]:
    """The body is the base pointer (u64 LE), already valid in this process — no mapping.

    FORK_SHM / FORK_COW: a host VA inherited across the fork, so the child already holds the same
    address (they differ in write semantics, not in how they resolve). DEVICE_MALLOC / VMM_WINDOW:
    a device pointer valid on the chip that allocated / carved it (the tensor must only reach that
    chip — a topology invariant). ``select_adapter`` enforces the owning-Worker half of that; the
    exact-chip half for a Worker with several chips is submit-time's job, not this cache's.
    """
    return int.from_bytes(desc.body, "little"), None


def _resolve_host_shm_map(desc: BufferDescriptor) -> tuple[int, SharedMemory | None]:
    """Map the named POSIX shm object into this process."""
    name = desc.body.decode("utf-8")
    try:
        shm = SharedMemory(name=name)
    except FileNotFoundError as exc:
        # The owner unlinks on release, so a missing name is the expected shape of "this identity
        # was released", not a corrupt descriptor. Naming both the identity and that reading
        # separates it from a genuinely bad name at a glance.
        raise FileNotFoundError(
            f"ImportRegistry: shm object {name!r} for {desc.identity} does not exist — its "
            f"owner has released the buffer, or it was never created"
        ) from exc
    # A mapping post-condition, not a capability: `validate_tensor` admits a view when
    # byte_offset + extent <= nbytes, so a backing smaller than the nbytes its own descriptor
    # advertises turns every one of those checks into a comparison against a number no memory
    # stands behind. The object's real size is the only thing here the owner cannot overstate.
    if shm.size < desc.nbytes:
        shm.close()
        raise ValueError(
            f"ImportRegistry: shm object {name!r} is {shm.size} bytes, short of the {desc.nbytes} its descriptor claims"
        )
    return _shm_base_addr(shm), shm


# One resolver per mechanism, so a mechanism shared by several backends is carried out once.
_RESOLVERS: dict[AdapterProfile, Callable[[BufferDescriptor], tuple[int, SharedMemory | None]]] = {
    AdapterProfile.FORK_INHERITED_VA: _resolve_body_as_va,
    AdapterProfile.DEVICE_LOCAL: _resolve_body_as_va,
    AdapterProfile.HOST_SHM_MAP: _resolve_host_shm_map,
}


@dataclass
class ImportedBuffer:
    """A buffer materialized into the consumer's address space: identity -> local base.

    ``profile`` names the materialization mechanism and ``capabilities`` names the operations this
    endpoint may perform through it.  They are deliberately separate from the wire
    ``BackendKind`` and from ``AdapterKind``: a backend is a producer-side representation, while a
    profile is the consumer-side mechanism and capabilities are the resulting access rights.
    """

    identity: CanonicalIdentity
    base: int
    nbytes: int
    address_space: AddressSpace = AddressSpace.HOST
    shm: SharedMemory | None = None  # the consumer's own mapping for shm backends
    # The descriptor this mapping was created from. Identity alone does not pin a backing: it says
    # WHICH allocation, not how big it is or how to reach it, so a second descriptor carrying the
    # same identity and a different nbytes / access / backend / body describes something the first
    # mapping is not. Keeping it is what lets `materialize` detect that instead of silently handing
    # back the earlier mapping.
    descriptor: BufferDescriptor | None = None
    profile: AdapterProfile | None = None
    capabilities: frozenset[BufferCapability] = frozenset()


@dataclass
class MappedArg:
    """A Python compute (sub-worker) task arg: a ``Tensor`` materialized into this process, exposing a
    ``buffer`` at the view origin plus the view geometry. The callable computes with e.g.
    ``torch.frombuffer(arg.buffer, dtype=<from arg.dtype>, count=prod(arg.shapes))`` — reads/writes
    land in the shared backing the owner sees, except when the descriptor's ``access`` is
    ``AccessMode.READ``, where ``buffer`` is a read-only view (a ``FORK_COW`` backing's writes are
    invisible to the owner and are the reason ``access`` is forced to ``READ`` for it).
    """

    imported: ImportedBuffer
    byte_offset: int
    shapes: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: int  # DataType value

    @property
    def buffer(self) -> memoryview:
        """A memoryview over the mapped backing at this view's origin (``byte_offset``); read-only
        when the descriptor's ``access`` is ``AccessMode.READ``."""
        ib = self.imported
        if ib.shm is not None:
            base = ib.shm.buf
            assert base is not None
        else:
            # FORK_SHM / FORK_COW: no shm object — the base is a host VA inherited across the
            # fork, so wrap that range directly.
            base = memoryview((ctypes.c_char * ib.nbytes).from_address(ib.base))
        view = base[self.byte_offset :]
        if ib.descriptor is not None and ib.descriptor.access == AccessMode.READ:
            return view.toreadonly()
        return view


class MappedArgs(Sequence):
    """A Python sub-worker's task args: the mapped tensor args plus the scalar args.

    Indexes and iterates as the tensor ``MappedArg`` list (``args[i].buffer``, ``len(args)``) — the
    common compute-leaf access — and additionally exposes the blob's scalars via ``scalar_count()`` /
    ``scalar(i)`` (uint64, in submission order), mirroring the owner-side ``TaskArgs`` scalar API.
    """

    __slots__ = ("_scalars", "_tensors")

    def __init__(self, tensors: list[MappedArg], scalars: tuple[int, ...]) -> None:
        self._tensors = list(tensors)
        self._scalars = tuple(int(s) for s in scalars)

    def __getitem__(self, i):
        return self._tensors[i]

    def __len__(self) -> int:
        return len(self._tensors)

    def tensor_count(self) -> int:
        return len(self._tensors)

    def scalar_count(self) -> int:
        return len(self._scalars)

    def scalar(self, i: int) -> int:
        return self._scalars[i]


@dataclass(frozen=True)
class ImportContext:
    """Which Worker's device memory this endpoint is related to, and how.

    ``device_owner_instance_id`` names that Worker; ``is_host_endpoint`` says which side of it this
    endpoint sits on, and those two answer every DEVICE-backing question ``select_adapter`` asks:

    ===================  =================  =========================================
    endpoint             is_host_endpoint   relation to a backing carrying that nonce
    ===================  =================  =========================================
    chip child           False              it IS one of that Worker's chips -> DIRECT_MAP
    the Worker itself    True               it OWNS those chips -> OWNER_DELEGATED_COPY
    Python SUB child     True (nonce None)  no relation at all -> refused
    ===================  =================  =========================================

    The two host rows are the distinction the nonce carries and the boolean cannot: neither can
    hold a device VA, and only one of them can still reach the backing, by driving the control
    mailbox of the chip that owns it. A nonce that does not match is a different Worker's device
    memory, refused from every endpoint here.

    This is Worker-grained, not chip-grained: the nonce is minted once per Worker incarnation (in
    ``Worker.init()``), not once per chip, so a Worker with several ``device_ids`` gives all its
    chip children the same one — the wire ``BufferDescriptor`` has no field that distinguishes
    siblings (``owner_worker_id`` is owner-side free/copy provenance, never serialized). A backing
    minted for chip 0 therefore also passes here on sibling chip 1. The exact-chip half is
    ``Worker._child_prov_check_dispatch``, which compares the registered handle's
    ``owner_worker_id`` against the submitted target; this is the backstop for a path that arrives
    without having gone through it, not a replacement for it.

    DEVICE-backing refusals are tagged ``RegionAccessReasonCode.UNSUPPORTED_ENDPOINT_RELATION``
    (``comm_endpoints.py``) — the same reason code the domain-scoped
    ``EndpointRegistry``/``RegionAccessService`` engine reports for the same question at
    region-planning time. The two mechanisms share only that vocabulary, not a live registry
    object: this context is built inside forked child processes (and at L2's same-process
    lazy-materialize point), both of which run before ``EndpointRegistry.from_snapshot()``'s
    ``_require_ready_for_region_planning()`` precondition can be assumed to hold. That registry's
    topology snapshot also carries no per-chip identity, so it could not close the
    Worker-grained-not-chip-grained limit above even where it is reachable.
    """

    is_host_endpoint: bool
    device_owner_instance_id: bytes | None = None


class ImportRegistry:
    """Per-consumer-endpoint lazy import cache: materialize a ``Tensor``'s embedded descriptor to a
    local base on first receipt (map-once), keyed by canonical identity.

    A consumer calls ``materialize`` for each tensor's embedded descriptor as it arrives; the first
    sight of an identity maps its backing into this process, later sights reuse the cached base
    (a bumped generation is a distinct identity, materialized fresh). Keyed by the canonical identity
    itself so lookups are exact — never a numeric-range guess, and never split by the wire padding
    the identity's equality and hashing deliberately ignore.

    Map-once is a cache, not a trust boundary: a cache hit re-checks that the incoming descriptor
    still describes the backing that was mapped, so one identity can never come to mean two things.
    """

    def __init__(self, context: ImportContext) -> None:
        self._by_identity: dict[CanonicalIdentity, ImportedBuffer] = {}
        self._context = context

    def materialize(self, desc: BufferDescriptor) -> ImportedBuffer:
        """Map ``desc``'s backing into this process on first sight of its identity; reuse the
        cached ImportedBuffer thereafter (map-once).

        Takes the descriptor, never its bytes: every producer of one — an owner's ``to_descriptor()``,
        a re-export, an element pulled out of a received TaskArgs — hands over the bound type, and
        there is no Python path from raw bytes to a descriptor at all, which is what keeps
        ``validate_buffer_descriptor`` a gate rather than a habit.

        Rejects a descriptor that conflicts with the one already mapped under its identity, and a
        POSIX shm object smaller than the ``nbytes`` its descriptor claims — every view bound check
        is computed against that number, so an unverified one makes them all vacuous.

        Selection is the capability judgment: ``select_adapter`` either names the mechanism that
        reaches this backing from this endpoint or refuses, and the resolver below only carries the
        named mechanism out. Nothing after selection looks at ``backend_kind`` again. A mechanism
        that reaches the backing without producing an address is refused *here* rather than there --
        this returns a mapping, and that is the only narrowing it applies to the judgment.
        """
        key = desc.identity
        cached = self._by_identity.get(key)
        if cached is not None:
            # Field-wise, so wire padding and the unused tail of `body` do not make one backing look
            # like two. The mapping that exists wins: it is the one already handed out, and callers
            # may hold addresses into it.
            if cached.descriptor is not None and cached.descriptor != desc:
                raise ValueError(
                    f"ImportRegistry: identity {desc.identity} is already materialized from a "
                    f"different descriptor ({_descriptor_delta(cached.descriptor, desc)}); one "
                    f"identity names one backing, so the mapping already handed out is kept and "
                    f"this is refused"
                )
            return cached
        kind, profile = select_adapter(desc, self._context)
        resolver = _RESOLVERS.get(profile)
        if resolver is None:
            raise _refuse(
                RegionAccessReasonCode.NO_IMPLEMENTED_DIRECT_MAP_PROBE,
                f"{desc.identity} is reachable from this endpoint by {kind.value} ({profile.value}), "
                f"which produces no local address; materialize returns a mapping, so ask "
                f"`select_adapter` what this endpoint may do and issue the operation through the "
                f"backing's owner",
            )
        base, shm = resolver(desc)
        imported = ImportedBuffer(
            desc.identity,
            base,
            desc.nbytes,
            desc.address_space,
            shm,
            desc,
            profile,
            capabilities_for_adapter(kind, desc.access),
        )
        self._by_identity[key] = imported
        return imported

    def materialize_args(self, args) -> dict[CanonicalIdentity, tuple[int, int]]:
        """Materialize every embedded descriptor in a ``TaskArgs`` and return the resolved map:
        identity -> (local base, address_space), scoped to this call's own tensors."""
        resolved: dict[CanonicalIdentity, tuple[int, int]] = {}
        for i in range(args.tensor_count()):
            desc = args.tensor(i).buffer
            imported = self.materialize(desc)
            resolved[desc.identity] = (imported.base, int(imported.address_space))
        return resolved

    def mapped_args_from_blob(self, blob_ptr: int, capacity: int) -> MappedArgs:
        """Materialize a task-args blob into a Python compute callable's args: every tensor becomes a
        MappedArg (map-once, buffer at the view origin) and the blob's scalars ride alongside. This is
        the compute-leaf map (a sub-worker reads/writes), distinct from pure forwarding (re-export,
        which never maps).
        """
        args = read_args_from_blob(blob_ptr, capacity)
        tensors = []
        for i in range(args.tensor_count()):
            t = args.tensor(i)
            if t.buffer.address_space == AddressSpace.DEVICE:
                # Depth behind the submit-time endpoint check: this process is a host compute leaf, so
                # a device address here would be handed to torch as a host pointer.
                raise ValueError(
                    f"sub-worker argument {i} is a DEVICE-space tensor "
                    f"({t.buffer.backend_kind.name}); it cannot be mapped into a host process"
                )
            tensors.append(MappedArg(self.materialize(t.buffer), t.byte_offset, t.shapes, t.strides, t.dtype))
        return MappedArgs(tensors, tuple(args.scalar(i) for i in range(args.scalar_count())))

    def resolve(self, identity: CanonicalIdentity) -> ImportedBuffer:
        """The already-materialized import for ``identity``. Raises ``KeyError`` if this endpoint has
        not materialized that backing — resolution never maps as a side effect."""
        return self.require(identity)

    def require(
        self,
        identity: CanonicalIdentity,
        need: BufferCapability | str | None = None,
    ) -> ImportedBuffer:
        """Return a live materialization, optionally requiring one capability.

        This is a pure lookup: it never maps as a side effect.  Missing identity and insufficient
        capability are intentionally different failures so callers can choose between materializing
        the backing and selecting a different operation adapter.
        """

        imported = self._by_identity.get(identity)
        if imported is None:
            raise KeyError(f"ImportRegistry: no buffer registered for {identity}")
        if need is None:
            return imported
        try:
            capability = BufferCapability(need)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ImportRegistry: unknown buffer capability {need!r}") from exc
        if capability not in imported.capabilities:
            raise ValueError(
                f"ImportRegistry: identity {identity} materialized with profile "
                f"{imported.profile!r} does not grant {capability.value}"
            )
        return imported

    def unregister(self, identity: CanonicalIdentity) -> None:
        """Drop ``identity``'s mapping if this endpoint made one; a no-op otherwise.

        Consumer-side only — unlinking belongs to the owning Worker, so this never destroys a
        backing, only this endpoint's own view of it. The owner broadcasts this on
        ``release_buffer()`` so a long-lived endpoint does not keep every backing it ever saw
        mapped for its entire lifetime; best-effort by design, since an endpoint that never
        materialized ``identity`` has nothing to drop.

        The entry is dropped only once its mapping is really gone. ``close()`` on a mapping whose
        consumer still holds a derived ``memoryview`` raises ``BufferError``, and an entry dropped
        before that point is a mapping nothing can reach to retry — so the raise leaves the entry
        in place for this registry's own ``close()`` to attempt again.
        """
        imported = self._by_identity.get(identity)
        if imported is None:
            return
        if imported.shm is not None:
            imported.shm.close()
        del self._by_identity[identity]

    def close(self) -> None:
        """Close every mapping this endpoint made. Consumer-side only — unlinking belongs to the
        owning Worker, so this never destroys a backing.

        Every mapping is attempted, and only the ones that closed are dropped: one endpoint holding
        an exported view must not strand every mapping behind it in the iteration order. The first
        error is raised once the sweep is done, so a caller still learns the endpoint leaked rather
        than seeing a silent success.
        """
        errors: list[BaseException] = []
        for identity, imported in list(self._by_identity.items()):
            if imported.shm is not None:
                try:
                    imported.shm.close()
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
                    continue
            del self._by_identity[identity]
        if errors:
            raise errors[0]
