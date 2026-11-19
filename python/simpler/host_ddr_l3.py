# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3-owned Host backing with endpoint-local A5 mappings."""

from __future__ import annotations

import ctypes
import os
import struct
import threading
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from enum import IntEnum

from _task_interface import (
    AddressSpace,
    BackendKind,
    TensorTransfer,
    _decode_buffer_descriptor_wire,
    _encode_buffer_descriptor_wire,
)


class _Action(IntEnum):
    QUIESCE = 2
    RELEASE = 3
    RECOVER = 4
    FINALIZE_RECOVERY = 5


_HEADER = struct.Struct("<4sBBHII")
_DESCRIPTOR_BYTES = 88
_live_parent_mappings: set[ParentHostMappings] = set()


def _check_descriptor(descriptor):
    if descriptor.address_space != AddressSpace.HOST or descriptor.backend_kind != BackendKind.POSIX_SHM:
        raise ValueError("L3 MAP requires ordinary shared Host backing")
    if not 0 < descriptor.nbytes <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("Host backing size must fit a nonzero uint64")


def _encode(action, rank, count, descriptor):
    return _HEADER.pack(b"HDDR", 3, int(action), 0, rank, count) + _encode_buffer_descriptor_wire(descriptor)


def _decode(payload):
    if len(payload) != _HEADER.size + _DESCRIPTOR_BYTES:
        raise ValueError("invalid Host mapping control length")
    magic, version, action, reserved, rank, count = _HEADER.unpack_from(payload)
    if magic != b"HDDR" or version != 3 or reserved:
        raise ValueError("invalid Host mapping control version")
    if not 1 <= count <= 65535 or rank >= count:
        raise ValueError("invalid Host mapping membership")
    descriptor = _decode_buffer_descriptor_wire(payload[_HEADER.size :])
    _check_descriptor(descriptor)
    return _Action(action), rank, count, descriptor


@dataclass
class _ParentRecord:
    buffer: object
    descriptor: object
    attempted: set[int] = field(default_factory=set)
    closing: bool = False


class ParentHostMappings:
    """Parent ownership persists until every attempted child confirms cleanup."""

    def __init__(self, *, members, alive, admission, lookup, dispatch, fanout):
        self._members = tuple(members)
        if not 1 <= len(self._members) <= 65535 or len(set(self._members)) != len(self._members):
            raise ValueError("L3 MAP requires distinct child workers")
        self._alive, self._admission = alive, admission
        self._lookup = lookup
        self._dispatch, self._fanout = dispatch, fanout
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._records = {}
        self._uses = {}

    def _check_pid(self):
        if self._pid != os.getpid():
            raise ValueError("Host mapping belongs to another process")

    def _check_context(self):
        self._check_pid()
        if not self._alive():
            raise ValueError("Host mapping belongs to an unavailable Worker context")

    def release_admission(self):
        return self._admission() if self.holds_resources else nullcontext()

    @property
    def holds_resources(self):
        with self._lock:
            return bool(self._records)

    def _phase(self, record, action):
        failures = {}

        def perform(member):
            try:
                payload = _encode(action, self._members.index(member), len(self._members), record.descriptor)
                if self._dispatch(member, payload) != payload:
                    raise RuntimeError("Host mapping control returned an invalid acknowledgement")
            except BaseException as exc:  # noqa: BLE001
                failures[member] = exc

        self._fanout(sorted(record.attempted), perform)
        if failures:
            raise RuntimeError(f"Host mapping {action.name} failed on children {sorted(failures)}") from next(
                iter(failures.values())
            )

    def _select(self, args):
        selected = {}
        for i in range(args.tensor_count()):
            tensor, mapped = args.tensor(i), args.transfer(i) == TensorTransfer.MAP
            if tensor.buffer.address_space != AddressSpace.HOST:
                continue
            if tensor.buffer.backend_kind != BackendKind.POSIX_SHM and not mapped:
                continue
            buffer = self._lookup(tensor.buffer.identity)
            if buffer is None or buffer.identity != tensor.buffer.identity:
                if mapped:
                    raise ValueError("MAP requires Host storage owned by this L3 Worker")
                continue
            identity = buffer.identity
            if identity in selected and selected[identity][1] != tensor.buffer:
                raise ValueError("Host arguments have conflicting backing descriptors")
            selected[identity] = buffer, tensor.buffer, mapped or (identity in selected and selected[identity][2])
        return selected

    def _validate_backing(self, buffer, descriptor):
        _check_descriptor(descriptor)
        if buffer.closed or buffer.to_descriptor() != descriptor:
            raise ValueError("Host descriptor differs from its original live backing")
        registration = buffer._mapping_registration
        if registration is not None and registration[0] is not self:
            raise ValueError("Host Buffer belongs to another mapping context")
        memory = buffer.shm.buf if buffer.shm is not None else None
        try:
            if memory is None or buffer.shm.size < descriptor.nbytes or buffer.shm.name.encode() != descriptor.body:
                raise ValueError("Host descriptor differs from its owned shared backing")
            if ctypes.addressof(ctypes.c_char.from_buffer(memory)) != buffer.base:
                raise ValueError("Host base differs from its shared backing")
        finally:
            if memory is not None:
                memory.release()

    def acquire(self, run_id, args, worker_id=None):
        """Retain backing and possible child cleanup before publishing task args."""
        self._check_context()
        if worker_id is not None and worker_id not in self._members:
            raise ValueError("MAP targets a worker outside its local membership")
        selected = self._select(args)
        with ExitStack() as locks:
            for buffer, _, _ in sorted(selected.values(), key=lambda value: id(value[0])):
                locks.enter_context(buffer._host_lock)
            with self._lock:
                for identity, (buffer, descriptor, _) in selected.items():
                    record = self._records.get(identity)
                    if record is not None and (record.closing or record.descriptor != descriptor):
                        raise ValueError("Host mapping is closing or has a conflicting descriptor")
                    self._validate_backing(buffer, descriptor)
                self._uses.setdefault(run_id, set()).update(selected)
                for identity, (buffer, descriptor, _) in selected.items():
                    record = self._records.get(identity)
                    if record is None:
                        record = _ParentRecord(buffer, descriptor)
                        self._records[identity] = record
                        buffer._mapping_registration = self, identity
                        _live_parent_mappings.add(self)
                    if worker_id is not None:
                        # Native publication can succeed even when submit raises.
                        record.attempted.add(worker_id)

    def release(self, run_id):
        self._check_pid()
        with self._lock:
            self._uses.pop(run_id, None)

    def check_unused(self, identity):
        self._check_pid()
        with self._lock:
            if any(identity in uses for uses in self._uses.values()):
                raise RuntimeError("Host Buffer has in-flight or unproven uses; wait before release")

    def _close_record(self, record, *, terminal=False):
        self.check_unused(record.descriptor.identity)
        record.closing = True
        if terminal:
            self._phase(record, _Action.RECOVER)
        # A lost reply can follow successful side effects. Keep every endpoint
        # obligation until an entire idempotent cleanup round is acknowledged.
        self._phase(record, _Action.QUIESCE)
        self._phase(record, _Action.RELEASE)
        if not terminal:
            del self._records[record.descriptor.identity]
            if not self._records:
                _live_parent_mappings.discard(self)

    def close(self, identity):
        self._check_pid()
        with self._lock:
            record = self._records.get(identity)
            if record is not None:
                self._close_record(record)

    def close_all(self, *, terminal=False):
        self._check_pid()
        with self._lock:
            if not self._records:
                return
        with self._admission(), self._lock:
            errors = []
            for record in list(self._records.values()):
                try:
                    self._close_record(record, terminal=terminal)
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)
            if errors:
                raise errors[0]
            if terminal:
                # All registrations must be gone before any child resets its
                # owner. Retain every record until finalization replies arrive.
                for record in self._records.values():
                    self._phase(record, _Action.FINALIZE_RECOVERY)
                self._records.clear()
                _live_parent_mappings.discard(self)

    def check_unregistered(self, identity):
        self._check_pid()
        with self._lock:
            if identity in self._records:
                raise RuntimeError("Host backing is registered or retains cleanup debt")

    def require_closed(self):
        self._check_pid()
        if self.holds_resources:
            raise RuntimeError("Host mappings still retain backing or cleanup debt")


@dataclass
class _ChildRecord:
    descriptor: object
    imported: object = None
    mapped_base: int = 0
    map_attempted: bool = False
    closing: bool = False


class ChildHostMappings:
    """Only coordinated parent cleanup may unregister a borrowed Host backing."""

    def __init__(self, chip, import_registry, *, owner_instance_id, rank, rank_count, alive):
        self._chip, self._imports = chip, import_registry
        self._owner, self._rank, self._rank_count = bytes(owner_instance_id), rank, rank_count
        self._alive, self._pid = alive, os.getpid()
        self._lock = threading.RLock()
        self._records, self._uses = {}, {}
        self._retired = set()
        self._terminal = False
        self._quiesced = False
        self._recovered = False
        self.native_finalized = False

    @property
    def pending_cleanup(self):
        with self._lock:
            return bool(self._records or self._uses)

    def _check_context(self):
        if self._pid != os.getpid() or not self._alive():
            raise ValueError("Host mapping belongs to an unavailable child context")
        if not self.native_finalized:
            self._chip._validate_execution_context()

    @property
    def holds_resources(self):
        with self._lock:
            return bool(self._records)

    def _recover(self, action):
        if action == _Action.RECOVER:
            # Only Worker.close sends this, after parent admissions/fences
            # drain. A task error alone is not a device last-use receipt.
            self._terminal = True
            if not self._quiesced:
                # Idle mappings can coexist with an unrelated failed run.
                # Ask the native owner even when this backing has no uses.
                self._recovered = bool(self._chip._quiesce_host_uses())
                self._quiesced = True
                self._uses.clear()
        else:
            if not self._terminal or self._records or self._uses:
                raise RuntimeError("Host recovery finalization requires all registrations and uses released")
            if self._recovered and not self.native_finalized:
                self._chip.finalize()
                self.native_finalized = True

    def handle(self, payload):
        action, rank, count, descriptor = _decode(payload)
        with self._lock:
            self._check_context()
            if rank != self._rank or count != self._rank_count or descriptor.identity.owner_instance_id != self._owner:
                raise ValueError("Host mapping control has foreign ownership or membership")
            identity = descriptor.identity
            record = self._records.get(identity)
            if record is not None and record.descriptor != descriptor:
                raise ValueError("Host mapping control conflicts with registered backing")
            if action in (_Action.RECOVER, _Action.FINALIZE_RECOVERY):
                self._recover(action)
                return payload
            if any(identity in uses for uses in self._uses.values()):
                raise RuntimeError("Host mapping has in-flight or unproven uses")
            if action == _Action.QUIESCE:
                self._retired.add(identity)
                if record is not None:
                    record.closing = True
            elif action == _Action.RELEASE:
                if record is not None:
                    if not record.closing:
                        raise RuntimeError("Host release requires coordinated quiescence")
                    if record.map_attempted:
                        self._chip._unmap_host_ddr(record.imported.base)
                        record.map_attempted = False
                        record.mapped_base = 0
                    del self._records[identity]
                try:
                    self._imports.unregister(identity)
                except BaseException:
                    if record is not None:
                        self._records[identity] = record
                    raise
                self._retired.add(identity)
            return payload

    def _prepare(self, descriptor):
        identity = descriptor.identity
        record = self._records.get(identity)
        if record is not None:
            if record.closing or not record.mapped_base:
                raise RuntimeError("Host mapping retains setup debt")
            return
        if not getattr(self._chip, "_host_ddr_mapping_supported", False):
            raise RuntimeError("MAP requires an A5 onboard runtime with Host mapping support")
        record = _ChildRecord(descriptor)
        self._records[identity] = record
        try:
            record.imported = self._imports.materialize(descriptor)
            record.map_attempted = True
            record.mapped_base = self._chip._map_host_ddr(record.imported.base, descriptor.nbytes)
            if not record.mapped_base:
                raise RuntimeError("Host mapping returned a null Device address")
        except BaseException:
            record.closing = True
            raise

    def _validate(self, descriptor):
        record = self._records.get(descriptor.identity)
        if record is None or record.closing or not record.mapped_base or record.descriptor != descriptor:
            raise ValueError("MAP requires a prepared live child mapping")
        return record

    def acquire(self, use_key, args):
        with self._lock:
            self._check_context()
            if self._terminal:
                raise RuntimeError("Host mapping child admission is terminally closed")
            if use_key in self._uses:
                raise RuntimeError("Host mapping use already reserved for this dispatch")
            selected = {}
            for i in range(args.tensor_count()):
                descriptor = args.tensor(i).buffer
                mapped = args.transfer(i) == TensorTransfer.MAP
                owned_host = (
                    descriptor.address_space == AddressSpace.HOST
                    and descriptor.backend_kind == BackendKind.POSIX_SHM
                    and descriptor.identity.owner_instance_id == self._owner
                )
                if not owned_host:
                    if mapped:
                        raise ValueError("MAP requires shared Host backing owned by this L3 Worker")
                    continue
                _check_descriptor(descriptor)
                identity = descriptor.identity
                if identity in self._retired:
                    raise ValueError("Host backing was already retired")
                if identity in selected and selected[identity][0] != descriptor:
                    raise ValueError("Host arguments have conflicting backing descriptors")
                record = self._records.get(identity)
                if record is not None and (record.closing or record.descriptor != descriptor):
                    raise ValueError("Host argument differs from its live registered backing")
                selected[identity] = descriptor, mapped or (identity in selected and selected[identity][1])
            if not selected:
                return
            # NONE/H2D uses can predate the first MAP of the same backing.
            self._uses[use_key] = frozenset(selected)
            try:
                for descriptor, mapped in selected.values():
                    if mapped:
                        self._prepare(descriptor)
            except BaseException:
                # No native task owns this dispatch yet; mapping debt stays cached.
                self._uses.pop(use_key, None)
                raise

    def resolve(self, tensor):
        with self._lock:
            self._check_context()
            record = self._validate(tensor.buffer)
            if not any(tensor.buffer.identity in uses for uses in self._uses.values()):
                raise ValueError("MAP address resolution requires an accepted use")
            return record.mapped_base, int(AddressSpace.DEVICE)

    def release(self, use_key):
        with self._lock:
            self._uses.pop(use_key, None)

    def check_unregistered(self, identity):
        with self._lock:
            if any(identity in uses for uses in self._uses.values()):
                raise RuntimeError("Host import has in-flight or unproven uses")
            if identity in self._records:
                raise RuntimeError("Host import is registered or retains cleanup debt")

    def require_closed(self):
        with self._lock:
            if self._records or self._uses:
                raise RuntimeError("Host mappings require coordinated parent release before child teardown")
