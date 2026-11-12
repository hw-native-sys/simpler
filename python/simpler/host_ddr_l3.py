# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3-owned Host backing with endpoint-local device mappings."""

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
    PREPARE = 1
    QUIESCE = 2
    RELEASE = 3


_HEADER = struct.Struct("<4sBBHQII")
_DESCRIPTOR_BYTES = 88
_live_mappings: set[ParentHostMappings | ChildHostMappings] = set()


def _attach_exception_note(error, note):
    notes = list(getattr(error, "__notes__", ()))
    notes.append(note)
    error.__notes__ = notes


def _check_descriptor(descriptor):
    if descriptor.address_space != AddressSpace.HOST or descriptor.backend_kind != BackendKind.POSIX_SHM:
        raise ValueError("L3 MAP requires ordinary shared Host backing")
    if not 0 < descriptor.nbytes <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("Host backing size must fit a nonzero uint64")


def _encode(action, group, rank, count, descriptor):
    return _HEADER.pack(b"HDDR", 2, int(action), 0, group, rank, count) + _encode_buffer_descriptor_wire(descriptor)


def _decode(payload):
    if len(payload) != _HEADER.size + _DESCRIPTOR_BYTES:
        raise ValueError("invalid Host mapping control length")
    magic, version, action, reserved, group, rank, count = _HEADER.unpack_from(payload)
    if magic != b"HDDR" or version != 2 or reserved or not 0 < group <= 0xFFFFFFFF:
        raise ValueError("invalid Host mapping control version or group")
    if not 1 <= count <= 65535 or rank >= count:
        raise ValueError("invalid Host mapping membership")
    descriptor = _decode_buffer_descriptor_wire(payload[_HEADER.size :])
    _check_descriptor(descriptor)
    return _Action(action), group, rank, count, descriptor


@dataclass
class _ParentRecord:
    buffer: object
    descriptor: object
    group: int
    attempted: set[int] = field(default_factory=set)
    ready: bool = False
    closing: bool = False
    deferred: bool = False


class ParentHostMappings:
    """Parent ownership persists until every attempted child confirms cleanup."""

    def __init__(self, *, members, alive, admission, allocate_group, lookup, dispatch, fanout, prepare_admission=None):
        self._members = tuple(members)
        if not 1 <= len(self._members) <= 65535 or len(set(self._members)) != len(self._members):
            raise ValueError("L3 MAP requires distinct child workers")
        self._alive, self._admission = alive, admission
        self._prepare_admission = prepare_admission if prepare_admission is not None else admission
        self._allocate_group, self._lookup = allocate_group, lookup
        self._dispatch, self._fanout = dispatch, fanout
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._records = {}
        self._uses = {}
        self._declared = {}

    def _check_pid(self):
        if self._pid != os.getpid():
            raise ValueError("Host mapping belongs to another process")

    def _check_context(self):
        self._check_pid()
        if not self._alive():
            raise ValueError("Host mapping belongs to an unavailable Worker context")

    def release_admission(self, identity):
        return self._admission() if self.is_registered(identity) else nullcontext()

    def is_registered(self, identity):
        with self._lock:
            return identity in self._records

    def is_deferred(self, identity):
        with self._lock:
            record = self._records.get(identity)
            return record is not None and record.deferred

    @property
    def holds_resources(self):
        with self._lock:
            return bool(self._records)

    def _phase(self, record, action):
        failures = {}
        members = sorted(record.attempted)

        def perform(member):
            try:
                payload = _encode(
                    action, record.group, self._members.index(member), len(self._members), record.descriptor
                )
                if self._dispatch(member, payload) != payload:
                    raise RuntimeError("Host mapping control returned an invalid acknowledgement")
            except BaseException as exc:  # noqa: BLE001
                failures[member] = exc

        self._fanout(members, perform)
        if action == _Action.RELEASE:
            record.attempted.difference_update(member for member in members if member not in failures)
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

    def prepare(self, args):
        self._check_context()
        selected = self._select(args)
        with self._lock:
            missing = any(mapped and identity not in self._records for identity, (_, _, mapped) in selected.items())
        # Cold control requires a drained tree. Reuse never takes this global gate.
        with ExitStack() as locks:
            if missing:
                locks.enter_context(self._prepare_admission())
            for buffer, _, _ in sorted(selected.values(), key=lambda value: id(value[0])):
                locks.enter_context(buffer._host_lock)
            with self._lock:
                for identity, (buffer, descriptor, _) in selected.items():
                    record = self._records.get(identity)
                    if record is not None and (record.closing or not record.ready or record.descriptor != descriptor):
                        raise ValueError("Host mapping is closing or has a conflicting descriptor")
                    self._validate_backing(buffer, descriptor)
                created = []
                try:
                    for identity, (buffer, descriptor, mapped) in selected.items():
                        if not mapped or identity in self._records:
                            continue
                        group = self._allocate_group()
                        if not 0 < group <= 0xFFFFFFFF:
                            raise ValueError("Host mapping group must be a nonzero uint32")
                        record = _ParentRecord(buffer, descriptor, group)
                        self._records[identity] = record
                        created.append(record)
                        buffer._mapping_registration = self, identity
                        _live_mappings.add(self)
                        record.attempted.update(self._members)
                        self._phase(record, _Action.PREPARE)
                        record.ready = True
                except BaseException as error:
                    for record in reversed(created):
                        try:
                            self._close_record(record)
                        except BaseException as cleanup:  # noqa: BLE001
                            _attach_exception_note(error, f"Host backing retained: rollback failed: {cleanup}")
                    raise
        return selected

    def acquire(self, run_id, args):
        self._check_context()
        selected = self._select(args)
        with ExitStack() as locks:
            for buffer, _, _ in sorted(selected.values(), key=lambda value: id(value[0])):
                locks.enter_context(buffer._host_lock)
            with self._lock:
                if run_id in self._uses:
                    raise RuntimeError("Host mapping use already reserved for this run")
                for identity, (buffer, descriptor, mapped) in selected.items():
                    if mapped:
                        self._validate(descriptor)
                    self._validate_backing(buffer, descriptor)
                self._uses[run_id] = frozenset(selected)
                self._declared[run_id] = frozenset(identity for identity, (_, _, mapped) in selected.items() if mapped)

    def _validate(self, descriptor):
        record = self._records.get(descriptor.identity)
        if record is None or record.closing or not record.ready or record.descriptor != descriptor:
            raise ValueError("MAP requires declared live Host backing prepared before the callback")
        if record.buffer.closed or record.buffer.to_descriptor() != descriptor:
            raise ValueError("stale Host backing descriptor")
        return record

    def validate_args(self, args, worker_id, run_id):
        self._check_context()
        with self._lock:
            if worker_id not in self._members:
                raise ValueError("MAP targets a worker outside its prepared membership")
            for i in range(args.tensor_count()):
                if args.transfer(i) == TensorTransfer.MAP:
                    descriptor = args.tensor(i).buffer
                    self._validate(descriptor)
                    if descriptor.identity not in self._declared.get(run_id, ()):
                        raise ValueError("MAP backing must be declared in the outer submission arguments")

    def release(self, run_id):
        self._check_pid()
        with self._lock:
            self._uses.pop(run_id, None)
            self._declared.pop(run_id, None)

    def check_unused(self, identity):
        self._check_pid()
        with self._lock:
            if any(identity in uses for uses in self._uses.values()):
                raise RuntimeError("Host Buffer has in-flight or unproven uses; wait before release")

    def _close_record(self, record):
        self.check_unused(record.descriptor.identity)
        record.closing = True
        # Only a confirmed RELEASE retires a child's obligation.
        self._phase(record, _Action.QUIESCE)
        self._phase(record, _Action.RELEASE)
        self._forget(record.descriptor.identity)

    def _forget(self, identity):
        self._records.pop(identity, None)
        if not self._records:
            _live_mappings.discard(self)

    def close(self, identity):
        self._check_pid()
        with self._lock:
            record = self._records.get(identity)
            if record is not None:
                self._close_record(record)

    def close_all(self, defer_unproven=None, unproven=()):
        self._check_pid()
        with self._lock:
            if not self._records:
                return
        with self._admission(), self._lock:
            errors = []
            for record in list(self._records.values()):
                if record.deferred:
                    continue
                if defer_unproven is not None and record.buffer.shm is not None:
                    record.buffer.shm.stop_exports()
                identity = record.descriptor.identity
                try:
                    if identity in unproven:
                        raise RuntimeError("Host backing has unproven run uses")
                    self._close_record(record)
                except BaseException as error:  # noqa: BLE001
                    if defer_unproven is not None:
                        try:
                            if identity in unproven:
                                raise error
                            # A live, quiesced child can retry a failed RELEASE.
                            self.check_unused(identity)
                            self._phase(record, _Action.QUIESCE)
                        except RuntimeError:
                            defer_unproven(record.buffer, record.attempted, error)
                            record.closing = True
                            record.deferred = True
                            continue
                    errors.append(error)
            if errors:
                raise errors[0]

    def check_unregistered(self, identity):
        self._check_pid()
        with self._lock:
            if identity in self._records:
                raise RuntimeError("Host backing is registered or retains cleanup debt")

    def retire_after_child_exit(self, identity):
        """Discard one deferred record after the owner has proved child cleanup."""
        with self._lock:
            for uses in (self._uses, self._declared):
                for run_id, identities in uses.items():
                    uses[run_id] = identities - {identity}
            self._forget(identity)


@dataclass
class _ChildRecord:
    group: int
    descriptor: object
    imported: object = None
    mapped_base: int = 0
    map_attempted: bool = False
    closing: bool = False


class ChildHostMappings:
    """Hold borrowed Host backings through coordinated release or drained shutdown."""

    def __init__(self, chip, import_registry, *, owner_instance_id, rank, rank_count, alive):
        self._chip, self._imports = chip, import_registry
        self._owner, self._rank, self._rank_count = bytes(owner_instance_id), rank, rank_count
        self._alive, self._pid = alive, os.getpid()
        self._lock = threading.RLock()
        self._records, self._uses = {}, {}
        self._retired_groups = set()
        self._cleanup_runs = {}

    def _check_context(self):
        if self._pid != os.getpid() or not self._alive():
            raise ValueError("Host mapping belongs to an unavailable child context")
        self._chip._validate_execution_context()

    @property
    def holds_resources(self):
        with self._lock:
            return bool(self._records)

    def handle(self, payload):
        action, group, rank, count, descriptor = _decode(payload)
        with self._lock:
            self._check_context()
            if rank != self._rank or count != self._rank_count or descriptor.identity.owner_instance_id != self._owner:
                raise ValueError("Host mapping control has foreign ownership or membership")
            identity = descriptor.identity
            record = self._records.get(identity)
            if record is not None and (record.group != group or record.descriptor != descriptor):
                raise ValueError("Host mapping control conflicts with registered backing")
            if action == _Action.PREPARE:
                self._prepare(group, descriptor, record)
                return payload
            if any(identity in uses for uses in self._uses.values()):
                raise RuntimeError("Host mapping has in-flight or unproven uses")
            if action == _Action.QUIESCE:
                if record is not None:
                    record.closing = True
            elif action == _Action.RELEASE:
                if record is not None:
                    if not record.closing:
                        raise RuntimeError("Host release requires coordinated quiescence")
                    self._release_record(identity, record)
                self._retired_groups.add(group)
            return payload

    def _release_record(self, identity, record):
        record.closing = True
        if record.map_attempted:
            self._chip._unmap_host_ddr(record.imported.base)
            record.map_attempted = False
            record.mapped_base = 0
        del self._records[identity]
        try:
            self._imports.unregister(identity)
        except BaseException:
            self._records[identity] = record
            raise
        if not self._records:
            _live_mappings.discard(self)

    def close_idle(self):
        with self._lock:
            self._check_context()
            for use_key, run in list(self._cleanup_runs.items()):
                if run._retry_owner_cleanup():
                    self.release(use_key)
            busy = set().union(*self._uses.values())
            errors = []
            for identity, record in list(self._records.items()):
                if identity not in busy:
                    try:
                        self._release_record(identity, record)
                    except BaseException as error:  # noqa: BLE001
                        errors.append(error)
            if errors:
                raise errors[0]

    def _prepare(self, group, descriptor, record):
        identity = descriptor.identity
        if group in self._retired_groups:
            raise ValueError("Host mapping group was already retired")
        if any(other.group == group and other.descriptor != descriptor for other in self._records.values()):
            raise ValueError("Host mapping group names conflicting backing")
        if record is not None:
            if record.closing or not record.mapped_base:
                raise RuntimeError("Host mapping retains setup debt")
            return
        if not getattr(self._chip, "_host_ddr_mapping_supported", False):
            raise RuntimeError("MAP requires an onboard runtime with Host mapping support")
        record = _ChildRecord(group, descriptor)
        self._records[identity] = record
        _live_mappings.add(self)
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
            if use_key in self._uses:
                raise RuntimeError("Host mapping use already reserved for this dispatch")
            identities = set()
            for i in range(args.tensor_count()):
                descriptor = args.tensor(i).buffer
                record = self._records.get(descriptor.identity)
                if args.transfer(i) == TensorTransfer.MAP:
                    record = self._validate(descriptor)
                if record is not None:
                    if record.closing or record.descriptor != descriptor:
                        raise ValueError("Host argument differs from its live registered backing")
                    identities.add(descriptor.identity)
            if identities:
                self._uses[use_key] = frozenset(identities)

    def resolve(self, tensor):
        with self._lock:
            self._check_context()
            record = self._validate(tensor.buffer)
            if not any(tensor.buffer.identity in uses for uses in self._uses.values()):
                raise ValueError("MAP address resolution requires an accepted use")
            return record.mapped_base, int(AddressSpace.DEVICE)

    def defer_release(self, use_key, run):
        with self._lock:
            if use_key in self._uses:
                self._cleanup_runs[use_key] = run

    def release(self, use_key):
        with self._lock:
            self._uses.pop(use_key, None)
            self._cleanup_runs.pop(use_key, None)

    def check_unregistered(self, identity):
        with self._lock:
            if identity in self._records:
                raise RuntimeError("Host import is registered or retains cleanup debt")

    def require_closed(self):
        with self._lock:
            if self._records or self._uses:
                raise RuntimeError("Host mappings require coordinated parent release before child teardown")
