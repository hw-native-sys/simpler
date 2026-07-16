# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3-L2 orchestrator communication facade."""

from __future__ import annotations

import ctypes
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from _task_interface import (  # pyright: ignore[reportMissingImports]
    _l3_host_mapped_counter_notify,
    _l3_host_mapped_counter_test,
    _l3_host_mapped_counter_wait,
    _l3_host_mapped_payload_read,
    _l3_host_mapped_payload_write,
    _l3_host_mapped_region_close,
)

from .task_interface import Tensor


class NotifyOp(IntEnum):
    Set = 0
    Add = 1


class WaitCmp(IntEnum):
    EQ = 0
    NE = 1
    GT = 2
    GE = 3
    LT = 4
    LE = 5


class _ServiceError(IntEnum):
    COPY_FAILED = 6
    SIGNAL_TIMEOUT = 7


class L3L2RegionAccessProfile(IntEnum):
    INVALID = 0
    ONBOARD_ACL_IPC = 1
    SIM_POSIX_SHM = 2


_MAX_SIGNED_CHRONO_TIMEOUT_NS = 2**63 - 1

_DESC = struct.Struct("<6Q")
_L3L2_ORCH_REGION_DESC_SCALAR_COUNT = 6
_ACL_IPC_EXPORT_KEY_BYTES = 65
_CTRL_SHM_TOKEN_BYTES = 32
_REGION_CREATE_REQUEST = struct.Struct("<QQQQi4x")
_REGION_CREATE_REPLY = struct.Struct(f"<6QIIi{_ACL_IPC_EXPORT_KEY_BYTES}s{_CTRL_SHM_TOKEN_BYTES}s3xQ")
_REGION_CREATE_REQUEST_BYTES = _REGION_CREATE_REQUEST.size
_REGION_CREATE_REPLY_BYTES = _REGION_CREATE_REPLY.size
_REGION_LAYOUT_ALIGNMENT = 64
_UINT64_MAX = (1 << 64) - 1
_REGION_MAGIC_VERSION = 0x4C334C3200020000


def _align_up(value: int, align: int) -> int:
    value = int(value)
    if value < 0 or value > _UINT64_MAX:
        raise ValueError("L3-L2 region layout overflowed uint64")
    remainder = value % align
    bump = 0 if remainder == 0 else align - remainder
    result = value + bump
    if result > _UINT64_MAX:
        raise ValueError("L3-L2 region layout overflowed uint64")
    return result


def _checked_add_u64(lhs: int, rhs: int) -> int:
    result = int(lhs) + int(rhs)
    if int(lhs) < 0 or int(rhs) < 0 or result > _UINT64_MAX:
        raise ValueError("L3-L2 region layout overflowed uint64")
    return result


@dataclass(frozen=True)
class L3L2OrchRegionDesc:
    magic_version: int
    region_id: int
    payload_base: int
    payload_bytes: int
    counter_base: int
    counter_bytes: int

    def scalars(self) -> list[int]:
        return [
            int(self.magic_version),
            int(self.region_id),
            int(self.payload_base),
            int(self.payload_bytes),
            int(self.counter_base),
            int(self.counter_bytes),
        ]


@dataclass(frozen=True)
class SignalTestResult:
    matched: bool
    observed: int


@dataclass(frozen=True)
class L3L2RegionCreateRequest:
    magic_version: int
    request_bytes: int
    payload_bytes: int
    counter_bytes: int
    l3_host_pid: int

    def encode_into(self, buf: memoryview, offset: int = 0) -> None:
        _REGION_CREATE_REQUEST.pack_into(
            buf,
            offset,
            int(self.magic_version),
            int(self.request_bytes),
            int(self.payload_bytes),
            int(self.counter_bytes),
            int(self.l3_host_pid),
        )


@dataclass(frozen=True)
class L3L2RegionCreateReply:
    desc: L3L2OrchRegionDesc
    access_profile: L3L2RegionAccessProfile
    device_id: int
    export_key: bytes
    backing_shm: str
    mapping_bytes: int


@dataclass
class L3HostRegionMapping:
    worker_id: int
    region_id: int
    access_profile: L3L2RegionAccessProfile
    total_bytes: int
    payload_offset: int
    payload_bytes: int
    counter_offset: int
    counter_bytes: int
    handle: int
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        try:
            _l3_host_mapped_region_close(int(self.handle))
        finally:
            self.closed = True


def decode_region_create_reply(buf: memoryview) -> L3L2RegionCreateReply:
    fields = _REGION_CREATE_REPLY.unpack_from(buf, 0)
    desc = L3L2OrchRegionDesc(*[int(v) for v in fields[:6]])
    access_profile = L3L2RegionAccessProfile(int(fields[6]))
    device_id = int(fields[8])
    export_key = bytes(fields[9]).split(b"\x00", 1)[0]
    backing_shm = bytes(fields[10]).split(b"\x00", 1)[0].decode("utf-8", "strict")
    return L3L2RegionCreateReply(
        desc=desc,
        access_profile=access_profile,
        device_id=device_id,
        export_key=export_key,
        backing_shm=backing_shm,
        mapping_bytes=int(fields[11]),
    )


def peek_region_create_reply_region_id(buf: memoryview) -> int:
    return int(_REGION_CREATE_REPLY.unpack_from(buf, 0)[1])


def validate_region_create_reply(
    reply: L3L2RegionCreateReply, expected_access_profile: L3L2RegionAccessProfile
) -> tuple[int, int]:
    desc = reply.desc
    if desc.magic_version != _REGION_MAGIC_VERSION:
        raise RuntimeError("create_l3_l2_region: reply magic_version is invalid")
    if desc.region_id == 0:
        raise RuntimeError("create_l3_l2_region: reply region_id must be nonzero")
    if reply.access_profile != expected_access_profile:
        raise RuntimeError(f"create_l3_l2_region: reply access_profile must be {expected_access_profile.name.lower()}")
    if desc.payload_bytes <= 0:
        raise RuntimeError("create_l3_l2_region: reply payload_bytes must be positive")
    if desc.counter_bytes <= 0 or desc.counter_bytes % 4 != 0:
        raise RuntimeError("create_l3_l2_region: reply counter_bytes must be positive and a multiple of 4")
    counter_offset = _align_up(desc.payload_bytes, _REGION_LAYOUT_ALIGNMENT)
    total_bytes = _checked_add_u64(counter_offset, desc.counter_bytes)
    expected_counter_base = _checked_add_u64(desc.payload_base, counter_offset)
    if desc.counter_base != expected_counter_base:
        raise RuntimeError("create_l3_l2_region: reply counter_base does not match fixed region layout")
    if desc.counter_base % _REGION_LAYOUT_ALIGNMENT != 0:
        raise RuntimeError("create_l3_l2_region: reply counter_base must be 64-byte aligned")
    if (
        reply.access_profile
        in (
            L3L2RegionAccessProfile.SIM_POSIX_SHM,
            L3L2RegionAccessProfile.ONBOARD_ACL_IPC,
        )
        and reply.mapping_bytes != total_bytes
    ):
        profile = reply.access_profile.name.lower()
        raise RuntimeError(f"create_l3_l2_region: {profile} reply mapping_bytes does not match descriptor layout")
    return counter_offset, total_bytes


class _PinnedBuffer:
    def __init__(self, obj: Any, *, writable: bool = False) -> None:
        self._keepalive: Any = obj
        if isinstance(obj, Tensor):
            if obj.child_memory:
                raise ValueError("L3-L2 payload buffer must be host storage, not child_memory device storage")
            if not obj.is_contiguous:
                raise ValueError("L3-L2 payload buffer must be contiguous")
            self.addr = int(obj.data)
            self.nbytes = int(obj.nbytes())
            return

        try:
            view = memoryview(obj)
        except TypeError as exc:
            raise ValueError("L3-L2 payload buffer must be a contiguous L3 Host-accessible byte span") from exc
        if not view.c_contiguous:
            raise ValueError("L3-L2 payload buffer must be contiguous")
        try:
            byte_view = view if view.itemsize == 1 and view.format in {"B", "b", "c"} else view.cast("B")
        except (TypeError, ValueError) as exc:
            raise ValueError("L3-L2 payload buffer must be viewable as bytes") from exc
        if writable and byte_view.readonly:
            raise ValueError("L3-L2 payload read destination must be writable")
        self.nbytes = int(byte_view.nbytes)
        if byte_view.readonly:
            staging = ctypes.create_string_buffer(byte_view.tobytes())
            self._keepalive = staging
            self.addr = ctypes.addressof(staging)
            return
        exported = ctypes.c_char.from_buffer(byte_view)
        self._keepalive = (byte_view, exported)
        self.addr = ctypes.addressof(exported)

    def close(self) -> None:
        return None

    def __enter__(self) -> _PinnedBuffer:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class L3L2OrchCounter:
    def __init__(self, region: L3L2OrchRegion, offset: int) -> None:
        self._region = region
        self._offset = int(offset)

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def addr(self) -> int:
        return int(self._region.descriptor.counter_base) + self._offset

    def notify(self, value: int, op: NotifyOp = NotifyOp.Set) -> None:
        self._region._ensure_live()
        op = NotifyOp(op)
        self._region._direct_counter_notify(self._offset, int(value), op)

    def test(self, cmp_value: int, cmp: WaitCmp) -> SignalTestResult:
        self._region._ensure_live()
        cmp = WaitCmp(cmp)
        return self._region._direct_counter_test(self._offset, int(cmp_value), cmp)

    def wait(self, cmp_value: int, cmp: WaitCmp, timeout: float) -> int:
        self._region._ensure_live()
        cmp = WaitCmp(cmp)
        if timeout is None or float(timeout) <= 0:
            raise ValueError("L3-L2 counter wait requires a positive timeout")
        timeout_s = float(timeout)
        timeout_ns = min(int(timeout_s * 1_000_000_000), _MAX_SIGNED_CHRONO_TIMEOUT_NS)
        return self._region._direct_counter_wait(self._offset, int(cmp_value), cmp, timeout_ns)


class L3L2OrchRegion:
    def __init__(
        self,
        owner: Any,
        worker_id: int,
        desc: L3L2OrchRegionDesc,
        l3_host_mapping: L3HostRegionMapping,
    ) -> None:
        self._owner = owner
        self._worker_id = int(worker_id)
        self._descriptor = desc
        self._l3_host_mapping = l3_host_mapping
        self._released = False
        self._poisoned = False
        self._expired = False

    @property
    def descriptor(self) -> L3L2OrchRegionDesc:
        return self._descriptor

    @property
    def region_id(self) -> int:
        return int(self._descriptor.region_id)

    def descriptor_scalars(self) -> list[int]:
        self._ensure_live()
        return self._descriptor.scalars()

    def payload_write(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        with _PinnedBuffer(host_buffer) as pinned:
            size = pinned.nbytes if nbytes is None else int(nbytes)
            self._validate_payload_range(offset, size, pinned.nbytes)
            try:
                _l3_host_mapped_payload_write(self._l3_host_mapping.handle, int(offset), pinned.addr, size)
            except Exception:
                self._poison()
                raise

    def payload_read(self, offset: int, host_buffer: Any, nbytes: int | None = None) -> None:
        self._ensure_live()
        with _PinnedBuffer(host_buffer, writable=True) as pinned:
            size = pinned.nbytes if nbytes is None else int(nbytes)
            self._validate_payload_range(offset, size, pinned.nbytes)
            try:
                _l3_host_mapped_payload_read(self._l3_host_mapping.handle, int(offset), pinned.addr, size)
            except Exception:
                self._poison()
                raise

    def counter(self, offset: int) -> L3L2OrchCounter:
        self._ensure_live()
        offset = int(offset)
        # Primitive validation is 4-byte; wrapper-owned writers still need separate 64-byte cache lines.
        if offset < 0 or offset % 4 != 0 or offset + 4 > int(self._descriptor.counter_bytes):
            raise ValueError("L3-L2 counter offset must be 4-byte aligned and inside the counter range")
        return L3L2OrchCounter(self, offset)

    def free(self) -> None:
        if self._released:
            return
        self._released = True

    def _expire(self) -> None:
        self._expired = True

    def _poison(self) -> None:
        self._poisoned = True

    def _ensure_live(self) -> None:
        if self._expired:
            raise RuntimeError(f"L3-L2 region {self.region_id} expired after orchestration run")
        if self._released:
            raise RuntimeError(f"L3-L2 region {self.region_id} has been released")
        if self._poisoned:
            raise RuntimeError(f"L3-L2 region {self.region_id} is poisoned")

    def _validate_payload_range(self, offset: int, nbytes: int, buffer_nbytes: int) -> None:
        offset = int(offset)
        nbytes = int(nbytes)
        if offset < 0 or nbytes <= 0:
            raise ValueError("L3-L2 payload offset must be non-negative and nbytes must be positive")
        if nbytes > int(buffer_nbytes):
            raise ValueError(f"L3-L2 payload nbytes={nbytes} exceeds host buffer size {buffer_nbytes}")
        payload_bytes = int(self._descriptor.payload_bytes)
        if offset + nbytes > payload_bytes:
            raise ValueError(f"L3-L2 payload range [{offset}, {offset + nbytes}) exceeds region size {payload_bytes}")

    def _close_l3_host_mapping(self) -> None:
        try:
            self._l3_host_mapping.close()
        except Exception:
            self._poison()
            raise

    def _direct_counter_notify(self, offset: int, value: int, op: NotifyOp) -> None:
        l3_host_mapping = self._l3_host_mapping
        mapping_offset = int(l3_host_mapping.counter_offset) + int(offset)
        try:
            _l3_host_mapped_counter_notify(l3_host_mapping.handle, mapping_offset, int(value), int(op))
        except Exception:
            self._poison()
            raise

    def _direct_counter_test(self, offset: int, cmp_value: int, cmp: WaitCmp) -> SignalTestResult:
        l3_host_mapping = self._l3_host_mapping
        mapping_offset = int(l3_host_mapping.counter_offset) + int(offset)
        try:
            matched, observed = _l3_host_mapped_counter_test(
                l3_host_mapping.handle, mapping_offset, int(cmp_value), int(cmp)
            )
        except Exception:
            self._poison()
            raise
        return SignalTestResult(matched=bool(matched), observed=int(observed))

    def _direct_counter_wait(self, offset: int, cmp_value: int, cmp: WaitCmp, timeout_ns: int) -> int:
        l3_host_mapping = self._l3_host_mapping
        mapping_offset = int(l3_host_mapping.counter_offset) + int(offset)
        try:
            status, error_kind, observed, _matched, message = _l3_host_mapped_counter_wait(
                l3_host_mapping.handle, mapping_offset, int(cmp_value), int(cmp), int(timeout_ns)
            )
        except Exception:
            self._poison()
            raise
        if int(status) == 0:
            return int(observed)
        msg = str(message) if message else "L3-L2 counter wait timed out"
        if int(error_kind) == int(_ServiceError.SIGNAL_TIMEOUT):
            raise TimeoutError(f"{msg}; observed={int(observed)}")
        self._poison()
        raise RuntimeError(f"{msg}; observed={int(observed)}")
