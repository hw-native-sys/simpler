# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Local Host registrations retained by accepted calls."""

from __future__ import annotations

import ctypes
import os
import threading
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from _task_interface import AddressSpace, BackendKind, BufferDescriptor, CanonicalIdentity, TensorTransfer

if TYPE_CHECKING:
    from .buffer import Buffer

_live_registries: set[HostMappingRegistry] = set()


@dataclass
class _Mapping:
    buffer: Buffer
    descriptor: BufferDescriptor
    host_base: int
    mapped_base: int = 0
    closing: bool = False
    registration_attempted: bool = False


class HostMappingRegistry:
    """Whole-Buffer registrations and run-scoped uses in one device context.

    Buffer locks precede the registry lock. Native registration is serialized by
    the registry lock; completion only removes a run's use set.
    """

    def __init__(self, chip: Any, alive: Callable[[], bool], lookup: Callable[[CanonicalIdentity], Any]):
        self._chip = chip
        self._alive = alive
        self._lookup = lookup
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._records: dict[CanonicalIdentity, _Mapping] = {}
        self._uses: dict[int, frozenset[CanonicalIdentity]] = {}

    def _check_pid(self):
        if os.getpid() != self._pid:
            raise ValueError("Host mapping belongs to another process")

    def _check_context(self):
        self._check_pid()
        if not self._alive():
            raise ValueError("Host mapping belongs to an unavailable device context")
        if self._chip is not None:
            self._chip._validate_execution_context()

    def _select(self, args):
        selected = {}
        for i in range(args.tensor_count()):
            tensor, transfer = args.tensor(i), args.transfer(i)
            if tensor.buffer.address_space != AddressSpace.HOST:
                continue
            buffer = self._lookup(tensor.buffer.identity)
            if buffer is None or buffer.identity != tensor.buffer.identity:
                if transfer == TensorTransfer.MAP:
                    raise ValueError("MAP requires Host storage owned by this L2 Worker")
                continue
            key = tensor.buffer.identity
            if key in selected and selected[key][1] != tensor.buffer:
                raise ValueError("Host arguments have conflicting backing descriptors")
            mapped = transfer == TensorTransfer.MAP or (key in selected and selected[key][2])
            selected[key] = (buffer, tensor.buffer, mapped)
        return selected

    def _validate_backing(self, buffer, descriptor, mapped):
        if buffer.closed or buffer.to_descriptor() != descriptor:
            raise ValueError("Host descriptor differs from its original live backing")
        if buffer.backend_kind != BackendKind.POSIX_SHM:
            raise ValueError("local Host mapping requires POSIX Host storage")
        if mapped and not getattr(self._chip, "_host_ddr_mapping_supported", False):
            raise RuntimeError("MAP requires an onboard runtime with Host mapping support")
        memory = buffer.shm.buf if buffer.shm is not None else None
        try:
            if memory is None or buffer.shm.size < descriptor.nbytes or buffer.shm.name.encode() != descriptor.body:
                raise ValueError("Host descriptor does not match its live shared-memory backing")
            if ctypes.addressof(ctypes.c_char.from_buffer(memory)) != buffer.base:
                raise ValueError("Host base differs from its owned shared-memory mapping")
        finally:
            if memory is not None:
                memory.release()

    def acquire(self, run_id, args):
        self._check_pid()
        selected = self._select(args)
        if not selected:
            return
        self._check_context()
        with ExitStack() as locks:
            for buffer, _, _ in sorted(selected.values(), key=lambda value: id(value[0])):
                locks.enter_context(buffer._host_lock)
            with self._lock:
                if run_id in self._uses:
                    raise RuntimeError("Host mapping use already reserved for this run")
                for identity, (buffer, descriptor, mapped) in selected.items():
                    record = self._records.get(identity)
                    if record is not None and (record.closing or record.descriptor != descriptor):
                        raise ValueError("Host mapping is closing or has a conflicting descriptor")
                    self._validate_backing(buffer, descriptor, mapped)
                created = []
                try:
                    self._uses[run_id] = frozenset(selected)
                    for identity, (buffer, descriptor, mapped) in selected.items():
                        record = self._records.get(identity)
                        if record is None:
                            record = _Mapping(buffer, descriptor, buffer.base)
                            created.append(identity)
                            self._records[identity] = record
                            buffer._mapping_registration = (self, identity)
                            _live_registries.add(self)
                        if mapped and not record.mapped_base:
                            record.registration_attempted = True
                            try:
                                record.mapped_base = self._chip._map_host_ddr(record.host_base, descriptor.nbytes)
                                if not record.mapped_base:
                                    raise RuntimeError("Host mapping returned a null address")
                            except BaseException:
                                record.closing = True
                                raise
                except BaseException as error:
                    self._uses.pop(run_id, None)
                    for identity in reversed(created):
                        record = self._records.get(identity)
                        if record is None:
                            continue
                        try:
                            self._close_locked(identity, record)
                        except BaseException as cleanup_error:
                            notes: list[str] = list(getattr(error, "__notes__", ()))
                            notes.append(f"Host mapping cleanup retained for retry: {cleanup_error}")
                            error.__notes__ = notes
                    raise

    def resolve(self, tensor):
        self._check_context()
        with self._lock:
            identity = tensor.buffer.identity
            record = self._records.get(identity)
            if record is None or record.descriptor != tensor.buffer:
                raise ValueError("MAP requires a prepared mapping in its owning L2 context")
            if record.closing or record.buffer.closed or not record.mapped_base:
                raise ValueError("Host mapping is closing or unavailable")
            if record.buffer.to_descriptor() != record.descriptor:
                raise ValueError("Host descriptor differs from its registered backing")
            if not any(identity in used for used in self._uses.values()):
                raise ValueError("MAP address resolution requires an accepted use")
            return record.mapped_base, int(AddressSpace.DEVICE)

    def release(self, run_id):
        self._check_pid()
        with self._lock:
            self._uses.pop(run_id, None)

    def release_after_context_close(self):
        """Discharge native Host uses after successful terminal context teardown."""
        self._check_pid()
        with self._lock:
            self._uses.clear()

    def close_idle(self):
        """Unregister unused cache entries while the native context is still alive.

        Outstanding or unproven uses retain their registration and backing.
        Buffer locks precede the registry lock, as in submission and release.
        """
        self._check_pid()
        with self._lock:
            records = tuple(self._records.items())
        for identity, record in records:
            with record.buffer._host_lock, self._lock:
                if self._records.get(identity) is not record:
                    continue
                if any(identity in used for used in self._uses.values()):
                    continue
                self._close_locked(identity, record)

    def check_unregistered(self, identity):
        self._check_pid()
        with self._lock:
            if identity in self._records:
                raise RuntimeError("Host backing is registered; use release_buffer after waiting for in-flight uses")

    def _check_unused_locked(self, identity):
        if any(identity in used for used in self._uses.values()):
            raise RuntimeError("Host Buffer has in-flight or unproven uses; wait before release")

    def check_unused(self, identity):
        self._check_pid()
        with self._lock:
            self._check_unused_locked(identity)

    def _close_locked(self, identity, record):
        self._check_unused_locked(identity)
        record.closing = True
        if record.registration_attempted:
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
