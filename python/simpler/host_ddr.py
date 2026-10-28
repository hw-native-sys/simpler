# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Owned Host mappings and use reservations for direct L2 submissions."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Callable

# A failed close must survive garbage collection: a Python cycle alone is not
# ownership when the device may still read its registered backing.
_live_sources: set[HostDdrSource] = set()


@dataclass
class _Mapping:
    buffer: Any
    descriptor: Any
    host_base: int
    mapped_base: int = 0


class _MappedHostMemory:
    """Preserve the buffer protocol view while fencing raw shm close/unlink."""

    def __init__(self, memory: Any, owner: HostDdrSource, identity: Any):
        self._memory = memory
        self._owner = owner
        self._identity = identity

    def __getattr__(self, name: str) -> Any:
        return getattr(self._memory, name)

    def close(self) -> None:
        self._owner.check_unregistered(self._identity)
        self._memory.close()

    def unlink(self) -> None:
        self._owner.check_unregistered(self._identity)
        self._memory.unlink()


class HostDdrSource:
    """A mapping never outlives its backing or loses an unproven accepted use."""

    def __init__(self, chip: Any, alive: Callable[[], bool]):
        self._chip = chip
        self._alive = alive
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._records: dict[Any, _Mapping] = {}
        self._uses: dict[int, frozenset[Any]] = {}

    def _check_context(self) -> None:
        if os.getpid() != self._pid:
            raise ValueError("device_readable Host buffer belongs to another process")
        if not self._alive():
            raise ValueError("device_readable Host buffer belongs to an unavailable device context")

    def register(self, buffer: Any) -> None:
        self._check_context()
        record = _Mapping(buffer, buffer.to_descriptor(), buffer.base)
        with self._lock:
            _live_sources.add(self)
            self._records[buffer.identity] = record
            buffer._source_registration = (self, buffer.identity)
            buffer.shm = _MappedHostMemory(buffer.shm, self, buffer.identity)
            record.mapped_base = self._chip._map_host_ddr(record.host_base, buffer.nbytes)

    def check_unregistered(self, identity: Any) -> None:
        if os.getpid() != self._pid:
            raise ValueError("Host mapping belongs to another process")
        with self._lock:
            if identity in self._records:
                raise RuntimeError("Host backing is registered; use release_buffer after waiting for in-flight uses")

    def validate(self, identity: Any, descriptor: Any) -> None:
        self._check_context()
        with self._lock:
            record = self._records.get(identity)
            if record is None:
                raise ValueError("device_readable Host buffer has been released")
            if record.descriptor != descriptor:
                raise ValueError("device_readable Host descriptor differs from its source registration")

    def acquire(self, run_id: int, args: Any) -> None:
        self._check_context()
        with self._lock:
            if run_id in self._uses:
                raise RuntimeError("Host mapping use already reserved for this run")
            selected = set()
            for i in range(args.tensor_count()):
                descriptor = args.tensor(i).buffer
                record = self._records.get(descriptor.identity)
                if record is None:
                    if descriptor.device_readable:
                        raise ValueError("device_readable Host buffer is not owned by this L2 context")
                    continue
                if descriptor != record.descriptor or not record.mapped_base:
                    raise ValueError("device_readable Host buffer has an invalid descriptor or mapping")
                selected.add(descriptor.identity)
            # One publication owns the complete set, including an interrupted return
            # to the submitter. Before native entry, rollback uses this same run id.
            self._uses[run_id] = frozenset(selected)

    def mapped_bases(self) -> dict[Any, int]:
        self._check_context()
        with self._lock:
            return {identity: record.mapped_base for identity, record in self._records.items()}

    def release(self, run_id: int) -> None:
        if os.getpid() != self._pid:
            raise ValueError("Host mapping use belongs to another process")
        with self._lock:
            self._uses.pop(run_id, None)

    def close(self, identity: Any) -> None:
        if os.getpid() != self._pid:
            raise ValueError("Host mapping belongs to another process")
        with self._lock:
            record = self._records.get(identity)
            if record is None:
                return
            if any(identity in identities for identities in self._uses.values()):
                raise RuntimeError("Host Buffer has in-flight or unproven uses; wait before release")
            self._chip._unmap_host_ddr(record.host_base)
            del self._records[identity]
            if not self._records:
                _live_sources.discard(self)
