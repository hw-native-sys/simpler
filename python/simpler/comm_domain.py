# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Communication-domain configuration and borrowed Host buffer mappings."""

from __future__ import annotations

import ctypes
import operator
import threading
import weakref
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, fields
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING

from _task_interface import AddressSpace

if TYPE_CHECKING:
    from .task_interface import CommBufferSpec


def normalize_workspace_engines(workspace_engines) -> tuple[str, ...]:
    if isinstance(workspace_engines, str) or workspace_engines is None:
        raise TypeError("workspace_engines must be a sequence of engine names")
    names = tuple(workspace_engines)
    if any(name not in ("sdma", "urma", "rdma") for name in names) or len(set(names)) != len(names):
        raise ValueError("workspace_engines requires distinct names from sdma, urma, rdma")
    workspace_engines = tuple(sorted(names))
    if "rdma" in workspace_engines and len(workspace_engines) > 1:
        raise ValueError("RDMA domains do not map peer windows; use separate domains for other engines")
    return workspace_engines


@dataclass(frozen=True)
class CommDomainRef:
    """Creator identity and generation of an already established domain."""

    owner_instance_id: bytes
    domain_id: int
    generation: int

    def __post_init__(self):
        if not isinstance(self.owner_instance_id, bytes) or len(self.owner_instance_id) != 8:
            raise ValueError("domain creator must be an 8-byte instance identity")
        if not any(self.owner_instance_id) or not 0 <= self.domain_id < 2**64 or not 0 < self.generation < 2**64:
            raise ValueError("invalid communication domain reference")


@dataclass(frozen=True)
class DomainConfig:
    members: tuple
    window_size: int
    buffers: tuple
    retain_after_run: bool
    topology: tuple
    workspace_engines: tuple[str, ...] = ()
    memory_source: AddressSpace = AddressSpace.DEVICE

    @classmethod
    def capture(
        cls,
        members,
        window_size,
        buffers,
        retain_after_run,
        topology,
        workspace_engines=(),
        memory_source=AddressSpace.DEVICE,
    ):
        snapshot = tuple(
            (
                str(b.name),
                str(b.dtype),
                operator.index(b.count),
                operator.index(b.nbytes),
                bool(b.load_from_host),
                bool(b.store_to_host),
            )
            for b in buffers
        )
        members = tuple(members)
        window_size = operator.index(window_size)
        if not members or len(members) > 64 or len(set(members)) != len(members):
            raise ValueError("domain members must contain 1 to 64 distinct chips")
        if window_size <= 0:
            raise ValueError("domain window_size must be positive")
        if window_size % 64:
            raise ValueError("domain window_size must be a multiple of 64 bytes")
        if len({b[0] for b in snapshot}) != len(snapshot) or any(not b[0] or b[3] <= 0 for b in snapshot):
            raise ValueError("domain buffers require unique names and positive nbytes")
        for name, _dtype, _count, nbytes, _load_from_host, _store_to_host in snapshot:
            if nbytes % 64:
                raise ValueError(
                    f"domain buffer {name!r}: nbytes={nbytes} must be a multiple of 64 bytes; "
                    "the caller must pad the physical buffer size and include it in window_size"
                )
        if sum(b[3] for b in snapshot) > window_size:
            raise ValueError("domain buffers exceed window_size")
        if memory_source not in (AddressSpace.HOST, AddressSpace.DEVICE):
            raise ValueError("domain memory_source must be AddressSpace.HOST or AddressSpace.DEVICE")
        workspace_engines = normalize_workspace_engines(workspace_engines)
        return cls(
            members, window_size, snapshot, bool(retain_after_run), tuple(topology), workspace_engines, memory_source
        )

    def workspace_mask(self) -> int:
        return sum(1 << ("sdma", "urma", "rdma").index(name) for name in self.workspace_engines)

    def specs(self) -> list[CommBufferSpec]:
        from .task_interface import CommBufferSpec  # noqa: PLC0415

        return [CommBufferSpec(*buffer) for buffer in self.buffers]

    def difference(self, other: DomainConfig) -> str:
        for item in fields(self):
            before, after = getattr(self, item.name), getattr(other, item.name)
            if before != after:
                return f"{item.name}: {before!r} != {after!r}"
        return ""


class HostBufferMapping(SharedMemory):
    """Borrow an L2-owned POSIX window through the standard SharedMemory API.

    L3 only closes its mapping. Creation and unlink belong to the native L2
    domain, so Python's resource tracker must not claim the backing.
    """

    def __init__(self, name: str, size: int):
        self._export_lock = threading.RLock()
        self._export_count = 0
        self._closing = False
        if not name or len(name) > 31 or not all(c.isascii() and (c.isalnum() or c in "_-") for c in name):
            raise ValueError("invalid Host domain shared memory name")
        try:
            super().__init__(name=name, create=False, track=False)
        except TypeError:  # Python < 3.13 has no per-instance tracking switch.
            super().__init__(name=name, create=False)
            resource_tracker.unregister(self._name, "shared_memory")
        if size <= 0 or self.size < size:
            self.close()
            raise ValueError("Host domain shared memory is smaller than the mapped window")

    def unlink(self) -> None:
        raise RuntimeError("Host domain shared memory belongs to L2; use domain.release()")

    @property
    def buf(self) -> memoryview | None:
        """Export the whole window, retaining a borrow through derived views."""
        with self._export_lock:
            if self._mmap is None:
                return None
            if self._closing:
                raise BufferError("Host domain mapping is closing; new CPU exports are forbidden")
            # NumPy can discard a temporary memoryview and keep only its base.
            # Keep the real mmap export in the carrier's finalizer, so derived
            # arrays pin it too. Explicit release orders the mmap unpin before
            # the borrow count drops, independent of ctypes deallocation order.
            backing = memoryview(self._mmap)
            address = ctypes.addressof(ctypes.c_ubyte.from_buffer(backing))
            carrier = (ctypes.c_ubyte * self.size).from_address(address)
            finalizer = weakref.finalize(carrier, self._release_export, backing)
            finalizer.atexit = False
            self._export_count += 1
            return memoryview(carrier).cast("B")

    def _release_export(self, backing: memoryview) -> None:
        with self._export_lock:
            backing.release()
            self._export_count -= 1

    def close(self) -> None:
        with self._export_lock:
            seal_host_mappings((self,))
            super().close()


def seal_host_mappings(mappings: Iterable[HostBufferMapping]) -> None:
    """Reject live exports, then prevent new ones across the entire release set.

    Nothing changes if any mapping is borrowed. Locks cover admission only;
    callers may wait for device fences and release native resources afterwards.
    """
    ordered = sorted({id(mapping): mapping for mapping in mappings}.values(), key=id)
    with ExitStack() as locks:
        for mapping in ordered:
            locks.enter_context(mapping._export_lock)
        for mapping in ordered:
            if mapping._export_count:
                raise BufferError(
                    f"Host domain mapping {mapping.name!r} has live CPU exports; "
                    "drop arrays and release memoryviews before closing or releasing the domain"
                )
        for mapping in ordered:
            mapping._closing = True
