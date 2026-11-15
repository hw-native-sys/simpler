# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Same-host VMM sharing with explicit consumer leases and caller-owned synchronization."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import _task_interface as _native
from _task_interface import _decode_buffer_descriptor_wire, _encode_buffer_descriptor_wire

if TYPE_CHECKING:
    from .buffer import Buffer, BufferDescriptor, Tensor

_CONTROL_FRAME = struct.Struct("<5Q")
# Native compute borrows use pipeline-slot IDs; external leases occupy the high-bit namespace.
_EXTERNAL_BORROW_TOKEN_BASE = 1 << 63


class _ControlOp(IntEnum):
    ALLOCATE = 0
    RETAIN = 1
    RELEASE = 2


@dataclass(frozen=True)
class _AllocationInfo:
    base: int
    shareable_handle: int
    mapping_bytes: int
    device_id: int
    cleanup: Buffer | None = None


@dataclass(frozen=True)
class BufferExport:
    """One consumer lease over the descriptor's backing identity and logical extent.

    The controller holds the source Worker alive and confirms release only after the
    consumer has drained external accesses, unregistered resources and unmapped the buffer.
    The shareable handle is valid on the same host; it is not a consumer device address.
    Sharing is for trusted processes: the native export disables PID whitelist checks.
    Reuse one consumer mapping per backing; a second import is subject to CANN constraints.
    """

    descriptor: bytes
    token: int
    device_id: int
    shareable_handle: int
    mapping_bytes: int


@dataclass
class _ExportLease:
    """Controller-local confirmation state for one outstanding export."""

    exported: BufferExport
    retain_confirmed: bool = False


class BufferExportError(RuntimeError):
    """An unconfirmed RETAIN; release_export(exported) confirms and retires it."""

    def __init__(self, exported: BufferExport) -> None:
        super().__init__(
            f"Buffer export {exported.token} is unconfirmed; retain the Worker until release_export succeeds"
        )
        self.exported = exported


class BufferImportError(RuntimeError):
    """A failed import whose mapping must be explicitly closed before releasing its export."""

    def __init__(self, mapping: BufferMapping) -> None:
        super().__init__("Buffer import cleanup is incomplete; close error.mapping in its importing ACL context")
        self.mapping = mapping


def _handle_command(chip: _native._ChipWorker, payload: bytes | memoryview) -> bytes:
    operation, base, size, token, _reserved = _CONTROL_FRAME.unpack(payload)
    if operation == _ControlOp.ALLOCATE:
        allocation = chip._malloc_shareable(base)
        return _CONTROL_FRAME.pack(
            allocation.device_addr, allocation.shareable_handle, allocation.mapping_bytes, chip.device_id, 0
        )
    if operation not in (_ControlOp.RETAIN, _ControlOp.RELEASE):
        raise ValueError(f"VMM control operation must be ALLOCATE, RETAIN or RELEASE, got {operation}")
    chip._vmm_borrow(base, size, token, operation == _ControlOp.RETAIN)
    return bytes(payload)


class BufferMapping:
    """Import into the caller's current ACL context, which must outlive this mapping.

    The controller retains both the source Worker and this object until all external
    users have finished. Object destruction is not a synchronization or release signal.
    """

    def __init__(self, exported: BufferExport) -> None:
        self.exported = exported
        self.descriptor: BufferDescriptor = _decode_buffer_descriptor_wire(exported.descriptor)
        if not 0 < self.descriptor.nbytes <= exported.mapping_bytes:
            raise ValueError(
                f"VMM logical extent {self.descriptor.nbytes} must be positive and fit mapping {exported.mapping_bytes}"
            )
        # The native object owns its registry entry; only ABI calls receive the integer handle ID.
        self._handle: _native._RegionHandle | None = None
        try:
            self._handle = _native._buffer_import_vmm(
                exported.device_id,
                exported.shareable_handle,
                exported.mapping_bytes,
                f"{bytes(self.descriptor.identity.owner_instance_id).hex()}:{exported.token}",
            )
        except RuntimeError as error:
            if not hasattr(error, "_buffer_mapping_handle"):
                raise
            self._handle = error._buffer_mapping_handle
            raise BufferImportError(self) from error

    @property
    def base(self) -> int:
        """Return this process's device VA while fully mapped in the importing context."""
        if self._handle is None:
            raise ValueError("VMM buffer mapping is closed")
        return int(_native._buffer_mapped_base(int(self._handle)))

    def address(self, tensor: Tensor) -> int:
        """Resolve a view of this backing after validating its logical extent."""
        if _encode_buffer_descriptor_wire(tensor.buffer) != self.exported.descriptor:
            raise ValueError(f"Tensor backing does not match buffer export token {self.exported.token}")
        view = self.descriptor.tensor(tuple(tensor.shapes), tensor.dtype, tuple(tensor.strides), tensor.byte_offset)
        return self.base + view.byte_offset

    def close(self) -> None:
        """Unmap after all external accesses and registrations end.

        A failed close retains unfinished native resources and the source lease. A
        partially unmapped buffer has no usable VA; retry resumes only unfinished cleanup.
        The controller may call owner.release_export only after this method succeeds.
        """
        if self._handle is not None:
            _native._buffer_close_vmm(int(self._handle))
            self._handle = None
