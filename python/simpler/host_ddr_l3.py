# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Collective Host mappings with parent-owned backing and endpoint-local authority."""

from __future__ import annotations

import ctypes
import os
import struct
import threading
from dataclasses import dataclass, field
from enum import IntEnum

from _task_interface import (
    AddressSpace,
    BackendKind,
    _decode_buffer_descriptor_wire,
    _encode_buffer_descriptor_wire,
)

from .host_ddr import _MappedHostMemory, _references


class _Action(IntEnum):
    MAP = 1
    PREPARE = 2
    ADD_VIEW = 3
    DROP_VIEW = 4
    RELEASE = 5
    QUIESCE = 6


_HEADER = struct.Struct("<4sBBHQII")
_VIEW = struct.Struct("<QHBB5I5I")
_DESCRIPTOR_BYTES = 88
_PAYLOAD_BYTES = _HEADER.size + _DESCRIPTOR_BYTES + _VIEW.size
_live_parent_mappings: set[ParentHostMappings] = set()


def _attach_exception_note(error: BaseException, note: str) -> None:
    adder = getattr(error, "add_note", None)
    if callable(adder):
        adder(note)
        return
    notes = getattr(error, "__notes__", None)
    if isinstance(notes, list):
        notes.append(note)
        return
    object.__setattr__(error, "__notes__", [note])


def _encode(action, group, rank, count, tensor):
    shapes = list(tensor.shapes)
    strides = list(tensor.strides)
    return (
        _HEADER.pack(b"HDDR", 1, int(action), 0, group, rank, count)
        + _encode_buffer_descriptor_wire(tensor.buffer)
        + _VIEW.pack(
            tensor.byte_offset,
            tensor._mapping_ref,
            tensor.ndims,
            int(tensor.dtype),
            *(shapes + [0] * (5 - len(shapes))),
            *(strides + [0] * (5 - len(strides))),
        )
    )


def _decode(payload):
    if len(payload) != _PAYLOAD_BYTES:
        raise ValueError("invalid Host mapping control length")
    magic, version, action, reserved, group, rank, count = _HEADER.unpack_from(payload)
    if magic != b"HDDR" or version != 1 or reserved or not 0 < group <= 0xFFFFFFFF:
        raise ValueError("invalid Host mapping control version or group")
    if not 2 <= count <= 65535 or rank >= count:
        raise ValueError("invalid Host mapping membership")
    descriptor = _decode_buffer_descriptor_wire(payload[_HEADER.size : _HEADER.size + _DESCRIPTOR_BYTES])
    offset, ref, dims, dtype, *geometry = _VIEW.unpack_from(payload, _HEADER.size + _DESCRIPTOR_BYTES)
    if not ref or not 1 <= dims <= 5 or any(geometry[dims:5]) or any(geometry[5 + dims :]):
        raise ValueError("invalid Host mapping view encoding")
    tensor = descriptor.tensor(geometry[:dims], dtype, strides=geometry[5 : 5 + dims], byte_offset=offset)
    tensor = tensor._with_mapping_ref(ref)
    _check_descriptor(descriptor)
    return _Action(action), group, rank, count, tensor


def _check_descriptor(descriptor):
    if descriptor.address_space != AddressSpace.HOST or descriptor.backend_kind != BackendKind.POSIX_SHM:
        raise ValueError("Host URMA requires ordinary shared Host backing")
    if not 0 < descriptor.nbytes <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("Host URMA backing size must fit a nonzero uint64")


@dataclass
class _ParentRecord:
    buffer: object
    descriptor: object
    group: int
    seed: object
    attempted: set[int] = field(default_factory=set)
    views: dict = field(default_factory=dict)
    closing: bool = False


class ParentHostMappings:
    """One complete Host backing exposed to every child, retained through cleanup debt.

    Callbacks retain the owning Worker and its transport. Admission must exclude
    accepted runs and forbid mapping changes inside orchestration callbacks.
    """

    def __init__(self, *, members, alive, admission, allocate_group, ensure_comm, dispatch, fanout):
        self._members = tuple(members)
        if not 2 <= len(self._members) <= 65535 or len(set(self._members)) != len(self._members):
            raise ValueError("Host URMA requires at least two distinct child workers")
        self._alive = alive
        self._admission = admission
        self._allocate_group = allocate_group
        self._ensure_comm = ensure_comm
        self._dispatch = dispatch
        self._fanout = fanout
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._record = None

    def _check_pid(self):
        if self._pid != os.getpid():
            raise ValueError("Host mapping belongs to another process")

    def _check_context(self):
        self._check_pid()
        if not self._alive():
            raise ValueError("Host mapping belongs to an unavailable Worker context")

    @property
    def holds_resources(self):
        with self._lock:
            return self._record is not None

    def _phase(self, record, action, tensor, members=None):
        targets = self._members if members is None else tuple(members)
        failures = {}

        def perform(member):
            try:
                payload = _encode(action, record.group, self._members.index(member), len(self._members), tensor)
                reply = self._dispatch(member, payload)
                if reply != payload:
                    raise RuntimeError("Host mapping control returned an invalid acknowledgement")
            except BaseException as exc:  # noqa: BLE001
                failures[member] = exc

        self._fanout(targets, perform)
        if failures:
            error = RuntimeError(f"Host mapping {action.name} failed on children {sorted(failures)}")
            raise error from next(iter(failures.values()))

    def map_view(self, buffer, tensor):
        with self._admission(), buffer._host_lock, self._lock:
            self._check_context()
            descriptor = buffer.to_descriptor()
            if tensor._mapping_ref or tensor.buffer != descriptor:
                raise ValueError("map_view requires an original live Host tensor")
            _check_descriptor(descriptor)
            memory = buffer.shm
            if memory is None or memory.size < descriptor.nbytes or memory.name.encode() != descriptor.body:
                raise ValueError("Host descriptor differs from its owned shared backing")
            if ctypes.addressof(ctypes.c_char.from_buffer(memory.buf)) != buffer.base:
                raise ValueError("Host Buffer base differs from its shared backing")
            record = self._record
            if record is not None and (record.closing or record.descriptor != descriptor):
                raise ValueError("another Host backing or cleanup debt is already registered")
            ref = next(_references)
            if ref > 65535:
                raise RuntimeError("Tensor mapping reference space exhausted")
            mapped = tensor._with_mapping_ref(ref)
            if record is None:
                if buffer._mapping_registration is not None and buffer._mapping_registration[0] is not self:
                    raise ValueError("Host Buffer belongs to another mapping context")
                group = self._allocate_group()
                if not 0 < group <= 0xFFFFFFFF:
                    raise ValueError("Host mapping group must be a nonzero uint32")
                record = _ParentRecord(buffer, descriptor, group, mapped)
                self._record = record
                buffer._mapping_registration = (self, buffer.identity)
                _live_parent_mappings.add(self)
                if not isinstance(buffer.shm, _MappedHostMemory):
                    buffer.shm = _MappedHostMemory(buffer.shm, self, buffer)
                try:
                    self._ensure_comm()
                    record.attempted.update(self._members)
                    self._phase(record, _Action.MAP, mapped)
                    self._phase(record, _Action.PREPARE, mapped)
                    self._phase(record, _Action.ADD_VIEW, mapped)
                except BaseException as exc:
                    self._rollback(record, exc)
                    raise
            else:
                try:
                    self._phase(record, _Action.ADD_VIEW, mapped)
                except BaseException as exc:
                    self._rollback(record, exc)
                    raise
            record.views[ref] = mapped
            return mapped

    def _rollback(self, record, original):
        record.closing = True
        try:
            self._close_record(record)
        except BaseException as cleanup:  # noqa: BLE001
            _attach_exception_note(original, f"Host backing retained: rollback failed: {cleanup}")

    def _validate(self, tensor, worker_id=None):
        self._check_context()
        record = self._record
        if worker_id is not None and worker_id not in self._members:
            raise ValueError("mapped Host tensor targets a worker outside its registration")
        if (
            record is None
            or record.closing
            or record.buffer.closed
            or tensor._mapping_ref not in record.views
            or record.views[tensor._mapping_ref] != tensor
            or record.buffer.to_descriptor() != record.descriptor
        ):
            raise ValueError("stale or foreign Host mapping reference, descriptor or range")
        return record

    def validate(self, tensor, worker_id=None):
        with self._lock:
            self._validate(tensor, worker_id)

    def validate_args(self, args, worker_id=None):
        with self._lock:
            for i in range(args.tensor_count()):
                tensor = args.tensor(i)
                if tensor._mapping_ref:
                    self._validate(tensor, worker_id)

    def unmap_view(self, tensor):
        with self._admission(), self._lock:
            self._check_pid()
            record = self._record
            if record is None or tensor._mapping_ref not in record.views or record.views[tensor._mapping_ref] != tensor:
                raise ValueError("stale or foreign Host mapping reference")
            if record.closing or len(record.views) == 1:
                self._close_record(record)
            else:
                try:
                    self._phase(record, _Action.DROP_VIEW, tensor)
                except BaseException as exc:
                    self._rollback(record, exc)
                    raise
                del record.views[tensor._mapping_ref]

    def _close_record(self, record):
        record.closing = True
        # RELEASE is idempotent. An exception says nothing about remote ownership;
        # keep every attempted endpoint until the whole release round is confirmed.
        # Revoke admission everywhere before any endpoint unregisters memory.
        # A failed quiescence check leaves all native registrations intact.
        self._phase(record, _Action.QUIESCE, record.seed, sorted(record.attempted))
        self._phase(record, _Action.RELEASE, record.seed, sorted(record.attempted))
        self._record = None
        _live_parent_mappings.discard(self)

    def close(self, identity):
        self._check_pid()
        with self._lock:
            if self._record is None or self._record.descriptor.identity != identity:
                return
        with self._admission(), self._lock:
            self._check_pid()
            record = self._record
            if record is not None and record.descriptor.identity == identity:
                self._close_record(record)

    def close_all(self):
        self._check_pid()
        with self._lock:
            if self._record is None:
                return
        with self._admission(), self._lock:
            self._check_pid()
            if self._record is not None:
                self._close_record(self._record)

    def check_unregistered(self, identity):
        self._check_pid()
        with self._lock:
            if self._record is not None and self._record.descriptor.identity == identity:
                raise RuntimeError("Host backing is registered or retains cleanup debt")

    def require_closed(self):
        self._check_pid()
        if self.holds_resources:
            raise RuntimeError("Host mappings still retain backing or cleanup debt")


@dataclass
class _ChildRecord:
    group: int
    descriptor: object
    imported: object = None
    mapped_base: int = 0
    map_attempted: bool = False
    workspace_attempted: bool = False
    ready: bool = False
    closing: bool = False
    views: dict = field(default_factory=dict)
    retired_refs: set[int] = field(default_factory=set)


class ChildHostMappings:
    """Borrowed shared backing; only a coordinated parent RELEASE tears it down."""

    def __init__(self, chip, import_registry, *, owner_instance_id, rank, rank_count, alive):
        self._chip = chip
        self._imports = import_registry
        self._owner = bytes(owner_instance_id)
        self._rank = rank
        self._rank_count = rank_count
        self._alive = alive
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._record = None
        self._uses = set()
        self._retired_groups = set()

    def _check_context(self):
        if self._pid != os.getpid() or not self._alive():
            raise ValueError("Host mapping belongs to an unavailable child context")

    @property
    def holds_resources(self):
        with self._lock:
            return self._record is not None

    def handle(self, payload, *, base_comm_handle):  # noqa: PLR0912 -- bounded control actions share one ownership gate
        action, group, rank, count, tensor = _decode(payload)
        with self._lock:
            self._check_context()
            if (
                rank != self._rank
                or count != self._rank_count
                or tensor.buffer.identity.owner_instance_id != self._owner
            ):
                raise ValueError("Host mapping control has foreign ownership or membership")
            record = self._record
            if record is not None and (record.group != group or record.descriptor != tensor.buffer):
                raise ValueError("Host mapping control conflicts with registered backing")
            if self._uses:
                raise RuntimeError("Host mapping has in-flight or unproven uses")
            if action == _Action.QUIESCE:
                if record is not None:
                    record.closing = True
                return payload
            if action == _Action.RELEASE:
                if record is not None and not record.closing:
                    raise RuntimeError("Host release requires coordinated quiescence")
                self._release_record(record)
                self._retired_groups.add(group)
                return payload
            if group in self._retired_groups:
                raise ValueError("Host mapping group was already retired")
            if action == _Action.MAP:
                if record is not None:
                    if record.closing or not record.mapped_base:
                        raise RuntimeError("Host mapping retains setup debt")
                    return payload
                if not getattr(self._chip, "_host_urma_supported", False):
                    raise RuntimeError("Host URMA requires an A5 onboard TMR runtime")
                record = _ChildRecord(group, tensor.buffer)
                self._record = record
                try:
                    record.imported = self._imports.materialize(tensor.buffer)
                    record.map_attempted = True
                    record.mapped_base = self._chip._map_host_ddr(record.imported.base, tensor.buffer.nbytes)
                    if not record.mapped_base:
                        raise RuntimeError("Host mapping returned a null Device address")
                except BaseException:
                    record.closing = True
                    raise
            elif record is None or record.closing or not record.mapped_base:
                raise ValueError("Host mapping control requires a prepared local import")
            elif action == _Action.PREPARE:
                if not base_comm_handle:
                    raise ValueError("Host URMA requires initialized child communicator")
                if not record.ready:
                    record.workspace_attempted = True
                    try:
                        workspace = self._chip._prepare_host_urma(
                            base_comm_handle, group, record.imported.base, record.descriptor.nbytes
                        )
                        if not workspace:
                            raise RuntimeError("Host URMA returned a null workspace")
                        record.ready = True
                    except BaseException:
                        record.closing = True
                        raise
            elif action == _Action.ADD_VIEW:
                if not record.ready:
                    raise ValueError("Host mapping workspace is not ready")
                if tensor._mapping_ref in record.retired_refs:
                    raise ValueError("Host mapping reference was already retired")
                old = record.views.get(tensor._mapping_ref)
                if old is not None and old != tensor:
                    raise ValueError("Host mapping reference names a different view")
                record.views[tensor._mapping_ref] = tensor
            elif action == _Action.DROP_VIEW:
                if tensor._mapping_ref not in record.views or record.views[tensor._mapping_ref] != tensor:
                    raise ValueError("stale or foreign Host mapping reference")
                del record.views[tensor._mapping_ref]
                record.retired_refs.add(tensor._mapping_ref)
            return payload

    def _release_record(self, record):
        if record is None:
            return
        record.closing = True
        if record.workspace_attempted:
            self._chip._release_host_urma()
            record.workspace_attempted = False
            record.ready = False
        if record.map_attempted:
            self._chip._unmap_host_ddr(record.imported.base)
            record.map_attempted = False
            record.mapped_base = 0
        self._record = None

    def _validate(self, tensor):
        self._check_context()
        record = self._record
        if (
            record is None
            or record.closing
            or not record.ready
            or tensor._mapping_ref not in record.views
            or record.views[tensor._mapping_ref] != tensor
        ):
            raise ValueError("stale or foreign Host mapping reference, descriptor or range")
        return record

    def resolve(self, tensor):
        with self._lock:
            return self._validate(tensor).mapped_base, int(AddressSpace.DEVICE)

    def acquire(self, use_key, args):
        with self._lock:
            self._check_context()
            if use_key in self._uses:
                raise RuntimeError("Host workspace use already reserved for this dispatch")
            for i in range(args.tensor_count()):
                tensor = args.tensor(i)
                if tensor._mapping_ref:
                    self._validate(tensor)
                elif self._record is not None and tensor.buffer.identity == self._record.descriptor.identity:
                    if tensor.buffer != self._record.descriptor:
                        raise ValueError("native Host argument differs from its registered backing")
            if self._record is not None:
                if self._record.closing or not self._record.ready:
                    raise RuntimeError("Host workspace has unresolved setup or cleanup debt")
                self._uses.add(use_key)

    def release(self, use_key):
        with self._lock:
            self._uses.discard(use_key)

    def check_unregistered(self, identity):
        with self._lock:
            if self._record is not None and self._record.descriptor.identity == identity:
                raise RuntimeError("Host import is registered or retains cleanup debt")

    def require_closed(self):
        with self._lock:
            if self._record is not None or self._uses:
                raise RuntimeError("Host mappings require coordinated parent release before child teardown")
