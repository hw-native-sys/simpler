# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Local A5 Host access records owned by the consumer ImportRegistry."""

from __future__ import annotations

import ctypes
import itertools
import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from _task_interface import AddressSpace, BackendKind, BufferDescriptor, CanonicalIdentity, Tensor

if TYPE_CHECKING:
    from .buffer import Buffer

# References are never recycled in a process, including across Worker generations.
_references = itertools.count(1)
_live_registries: set[HostMappingRegistry] = set()


@dataclass
class _Mapping:
    buffer: Buffer
    descriptor: BufferDescriptor
    host_base: int
    mapped_base: int = 0
    closing: bool = False
    views: dict[int, Tensor] = field(default_factory=dict)


class _MappedHostMemory:
    """Keep legacy shm access under the mapping's backing lifetime guard."""

    def __init__(self, memory, registry, buffer):
        self._memory = memory
        self._registry = registry
        self._buffer = buffer
        self._identity = buffer.identity

    def __getattr__(self, name):
        return getattr(self._memory, name)

    def close(self):
        self._registry._check_pid()
        with self._buffer._host_lock:
            self._registry.check_unregistered(self._identity)
            self._memory.close()

    def unlink(self):
        self._registry._check_pid()
        with self._buffer._host_lock:
            self._registry.check_unregistered(self._identity)
            self._memory.unlink()


class HostMappingRegistry:
    """Synchronous Device access to ordinary Host backing.

    Cleanup debt retains storage and the loaded backend, but grants no new access.
    An accepted run reserves all references atomically before native admission.
    """

    def __init__(self, chip: Any, alive: Callable[[], bool]):
        self._chip = chip
        self._alive = alive
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._records: dict[CanonicalIdentity, _Mapping] = {}
        self._uses: dict[int, frozenset[tuple[CanonicalIdentity, int]]] = {}

    def _check_pid(self):
        if os.getpid() != self._pid:
            raise ValueError("Host mapping belongs to another process")

    def _check_context(self):
        self._check_pid()
        if not self._alive():
            raise ValueError("Host mapping belongs to an unavailable device context")

    def map_view(self, buffer, tensor):
        self._check_context()
        with buffer._host_lock:
            return self._map_view_locked(buffer, tensor)

    def _map_view_locked(self, buffer, tensor):
        self._check_context()
        if tensor._mapping_ref:
            raise ValueError("map_view requires a native Tensor")
        descriptor = buffer.to_descriptor()
        if descriptor != tensor.buffer or buffer.address_space != AddressSpace.HOST:
            raise ValueError("map_view requires the original live Host backing")
        if buffer.backend_kind != BackendKind.POSIX_SHM:
            raise ValueError("local Host mapping requires POSIX Host storage")
        if not getattr(self._chip, "_host_ddr_mapping_supported", False):
            raise RuntimeError("map_view requires an A5 onboard runtime with Host mapping support")
        memory = buffer.shm.buf if buffer.shm is not None else None
        if memory is None or buffer.shm.size < descriptor.nbytes or buffer.shm.name.encode() != descriptor.body:
            raise ValueError("Host descriptor does not match its live shared-memory backing")
        host_base = ctypes.addressof(ctypes.c_char.from_buffer(memory))
        if host_base != buffer.base:
            raise ValueError("Host base differs from its owned shared-memory mapping")
        with self._lock:
            ref = next(_references)
            if ref > 65535:
                raise RuntimeError("local Tensor mapping reference space exhausted")
            record = self._records.get(buffer.identity)
            if record is None:
                record = _Mapping(buffer, descriptor, host_base)
                buffer._mapping_registration = (self, buffer.identity)
                self._records[buffer.identity] = record
                try:
                    _live_registries.add(self)
                    if not isinstance(buffer.shm, _MappedHostMemory):
                        buffer.shm = _MappedHostMemory(buffer.shm, self, buffer)
                    record.mapped_base = self._chip._map_host_ddr(record.host_base, buffer.nbytes)
                    if not record.mapped_base:
                        raise RuntimeError("Host mapping returned a null address")
                except BaseException:
                    record.closing = True
                    raise
            if record.closing or record.descriptor != descriptor:
                raise ValueError("Host mapping is closing or has a conflicting descriptor")
            mapped = tensor._with_mapping_ref(ref)
            record.views[ref] = mapped
            return mapped

    def _validate_locked(self, tensor):
        record = self._records.get(tensor.buffer.identity)
        if record is None or tensor._mapping_ref not in record.views or record.views[tensor._mapping_ref] != tensor:
            raise ValueError("stale or foreign Host mapping reference, descriptor or range")
        if record.closing or record.buffer.closed or not record.mapped_base:
            raise ValueError("Host mapping is closing or unavailable")
        if record.buffer.to_descriptor() != record.descriptor:
            raise ValueError("Host descriptor differs from its registered backing")
        return record

    def resolve(self, tensor):
        self._check_context()
        with self._lock:
            record = self._validate_locked(tensor)
            return record.mapped_base, int(AddressSpace.DEVICE)

    def acquire(self, run_id, args):
        self._check_context()
        with self._lock:
            if run_id in self._uses:
                raise RuntimeError("Host mapping use already reserved for this run")
            selected = set()
            for i in range(args.tensor_count()):
                tensor = args.tensor(i)
                identity = tensor.buffer.identity
                if tensor._mapping_ref:
                    self._validate_locked(tensor)
                    selected.add((identity, tensor._mapping_ref))
                elif identity in self._records:
                    record = self._records[identity]
                    if record.buffer.closed or tensor.buffer != record.descriptor:
                        raise ValueError("native Host argument differs from its registered backing")
                    selected.add((identity, 0))
            self._uses[run_id] = frozenset(selected)

    def release(self, run_id):
        self._check_pid()
        with self._lock:
            self._uses.pop(run_id, None)

    def check_unregistered(self, identity):
        self._check_pid()
        with self._lock:
            if identity in self._records:
                raise RuntimeError("Host backing is registered; unmap its views before raw close/unlink")

    def unmap_view(self, tensor):
        self._check_pid()
        with self._lock:
            record = self._records.get(tensor.buffer.identity)
            if record is None or tensor._mapping_ref not in record.views or record.views[tensor._mapping_ref] != tensor:
                raise ValueError("stale or foreign Host mapping reference")
            identity = tensor.buffer.identity
            if any((identity, tensor._mapping_ref) in refs or (identity, 0) in refs for refs in self._uses.values()):
                raise RuntimeError("Host view has in-flight or unproven uses; wait before unmap")
            if len(record.views) > 1:
                del record.views[tensor._mapping_ref]
            else:
                self._close_locked(tensor.buffer.identity, record)

    def _close_locked(self, identity, record):
        if any(any(backing == identity for backing, _ in refs) for refs in self._uses.values()):
            raise RuntimeError("Host Buffer has in-flight or unproven uses; wait before release")
        record.closing = True
        self._chip._unmap_host_ddr(record.host_base)
        del self._records[identity]
        if not self._records:
            _live_registries.discard(self)

    def close(self, identity):
        self._check_pid()
        with self._lock:
            record = self._records.get(identity)
            if record is not None:
                self._close_locked(identity, record)

    def require_closed(self):
        self._check_pid()
        with self._lock:
            if self._records:
                raise RuntimeError("Host mappings still retain backing or cleanup debt")
